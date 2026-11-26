import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from flask import Flask, abort, render_template, request, send_file, send_from_directory
from io import BytesIO
from werkzeug.exceptions import RequestEntityTooLarge
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Script1.pdf_utils import has_extractable_pdf_text

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = PROJECT_ROOT / "Web" / "uploads"
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}
UPLOAD_RETENTION_SECONDS = 24 * 60 * 60
UPLOAD_NAME_PATTERN = re.compile(r"upload_[0-9a-f]{32}\.(?:png|jpe?g|bmp|tiff?|webp|pdf)\Z")
UPLOAD_CLEANUP_PATTERN = re.compile(r"upload_[a-zA-Z0-9_-]+\.(?:png|jpe?g|bmp|tiff?|webp|pdf)\Z")
PROCESSING_JOBS = {}
PROCESSING_JOBS_LOCK = threading.Lock()
OCR_PROCESS_LOCK = threading.Lock()
OCR_ACCEPTANCE_THRESHOLD = 0.90
JOB_RETENTION_SECONDS = 60 * 60
OCR_WORKER_TIMEOUT_SECONDS = 120
OCR_WORKER_MAX_TIMEOUT_SECONDS = 1800
OCR_SECONDS_PER_SCANNED_PAGE = 120
VLM_SERVICE_TIMEOUT_SECONDS = 100
APP_BUILD_ID = "auto-pdf-text-v7"
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def ocr_worker_timeout(source_path: Path) -> int:
    if source_path.suffix.lower() != ".pdf":
        return OCR_WORKER_TIMEOUT_SECONDS

    with pymupdf.open(source_path) as document:
        scanned_pages = sum(
            not has_extractable_pdf_text(
                page, page.get_text("text", sort=True).strip()
            )
            for page in document
        )
    if not scanned_pages:
        return OCR_WORKER_TIMEOUT_SECONDS
    return min(
        OCR_WORKER_MAX_TIMEOUT_SECONDS,
        OCR_WORKER_TIMEOUT_SECONDS
        + scanned_pages * OCR_SECONDS_PER_SCANNED_PAGE,
    )


def summarize_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "No text could be extracted from the document."
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return text[:300]
    if len(sentences) <= 3:
        return ". ".join(sentences)
    return ". ".join(sentences[:3]) + "."


def generate_starter_gcode(text: str) -> str:
    gcode_lines = [
        "; CNC Smart Writer starter pen plotter program",
        "; Human-verified text preparation - motion placeholders only",
        "G21 ; millimeters",
        "G90 ; absolute positioning",
        "G0 Z5 ; pen up",
    ]
    for line_number, line in enumerate(text.splitlines() or [""], start=1):
        y = (line_number - 1) * 10
        safe_comment = "".join(
            char for char in line[:80] if char.isprintable()
        )
        gcode_lines.extend([
            f"; Text line {line_number}: {safe_comment}",
            f"G0 X0 Y{y}",
            "G1 Z0 ; pen down",
            "G1 X10 ; placeholder motion",
            "G0 Z5 ; pen up",
        ])
    gcode_lines.extend(["G0 X0 Y0", "M30 ; end program"])
    return "\n".join(gcode_lines) + "\n"


def run_ocr_isolated(source_path: Path, languages: list[str]) -> dict:
    languages = ["en"]
    output_path = source_path.with_suffix(".ocr.json")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "Script1" / "script1.py"),
        str(source_path),
        "--output",
        str(output_path),
        "--ocr",
        "auto",
        "--lang",
        *languages,
    ]
    worker_env = os.environ.copy()
    worker_env["PYTHONIOENCODING"] = "utf-8"
    timeout_seconds = ocr_worker_timeout(source_path)

    try:
        for attempt in range(2):
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(PROJECT_ROOT),
                    env=worker_env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"OCR worker timed out after {timeout_seconds} seconds. "
                    "The document may contain many scanned pages or complex layouts."
                ) from exc

            if completed.returncode == 0:
                break

            exit_code = completed.returncode & 0xFFFFFFFF
            details = ANSI_ESCAPE_PATTERN.sub(
                "", completed.stderr or completed.stdout
            ).strip()
            if exit_code == 0xC0000005 and attempt == 0:
                app.logger.warning(
                    "OCR worker crashed with Windows access-violation code 0xC0000005; retrying once."
                )
                continue

            app.logger.error(
                "OCR worker failed with exit code 0x%08X. Details: %s",
                exit_code,
                details[-4000:] or "No diagnostic output was returned.",
            )
            if exit_code == 0xC0000005:
                raise RuntimeError(
                    "PaddleOCR's native CPU runtime crashed twice "
                    "(Windows access violation 0xC0000005). The web server is still running."
                )
            raise RuntimeError(
                f"OCR worker exited with code {completed.returncode}. "
                f"{details[-1000:] or 'No diagnostic output was returned.'}"
            )

        if completed.returncode != 0:
            raise RuntimeError("OCR worker failed after retry.")
        if not output_path.is_file():
            raise RuntimeError("OCR worker finished without creating its JSON result.")
        with output_path.open("r", encoding="utf-8") as result_file:
            return json.load(result_file)
    finally:
        if output_path.exists():
            output_path.unlink()


def call_vlm_service(
    text: str, task: str = "summary", ocr_confidence: float = 1.0
) -> dict:
    payload = json.dumps({
        "text": text,
        "task": task,
        "language": "en",
        "ocr_confidence": ocr_confidence,
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:8001/analyze",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            req, timeout=VLM_SERVICE_TIMEOUT_SECONDS
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        app.logger.warning("VLM service request failed: %s", exc)
        return {
            "status": "error",
            "result": f"VLM service is not available right now. {exc}",
            "source": "fallback"
        }


def cleanup_expired_uploads(upload_dir: Path) -> None:
    cutoff = time.time() - UPLOAD_RETENTION_SECONDS
    for path in upload_dir.iterdir():
        if not UPLOAD_CLEANUP_PATTERN.fullmatch(path.name):
            continue
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            app.logger.warning("Could not remove expired preview file: %s", path)


def update_job(job_id: str, **updates) -> None:
    with PROCESSING_JOBS_LOCK:
        job = PROCESSING_JOBS.get(job_id)
        if job is not None:
            job.update(updates)
            job["updated_at"] = time.time()


def build_result_context(
    result: dict,
    filename: str,
    suffix: str,
    upload_path: Path,
    job_id: str,
    language_label: str,
) -> dict:
    extracted_text = result.get("text", "")
    pages = result.get("pages", [])
    ocr_pages = [
        page for page in pages
        if page.get("confidence_source") != "pdf_text"
        and page.get("backend") != "pdf_text"
    ]
    page_confidences = [
        float(page.get("confidence", 0.0))
        for page in ocr_pages
    ]
    has_ocr_confidence = bool(page_confidences)
    ocr_confidence = min(page_confidences) if has_ocr_confidence else None
    confidence_percent = (
        max(0, min(100, round(ocr_confidence * 100)))
        if has_ocr_confidence else None
    )
    meets_target = bool(extracted_text) and all(
        confidence >= OCR_ACCEPTANCE_THRESHOLD for confidence in page_confidences
    )

    if not has_ocr_confidence and any(page.get("backend") == "pdf_text" for page in pages):
        confidence_label = "PDF text extracted directly — no OCR score"
    elif meets_target:
        confidence_label = "Model score meets the 90% target — human review is still required"
    else:
        confidence_label = "Below the 90% target — careful review required"

    if result.get("ocr_status") == "no_text_detected":
        ai_result = (
            f"OCR did not extract text from \"{filename}\". "
            f"{result.get('warning') or 'Check the OCR backend setup and the server console for details.'}"
        )
        model_status = "skipped"
    elif not meets_target:
        ai_result = (
            "AI summary was skipped because OCR confidence is below 90%. "
            "Review and correct the extracted text before approving it."
        )
        model_status = "skipped"
    else:
        ai_result = "OCR is complete. Preparing the model summary in the background..."
        model_status = "pending"

    return {
        "filename": filename,
        "job_id": job_id,
        "language_label": language_label,
        "page_count": result.get("page_count", 0),
        "summary": summarize_text(extracted_text),
        "ai_result": ai_result,
        "model_status": model_status,
        "model_input_truncated": False,
        "raw_text": extracted_text,
        "ocr_warning": result.get("warning"),
        "ocr_status": result.get("ocr_status", "unknown"),
        "pages": pages,
        "backend": pages[0].get("backend", "unknown") if pages else "unknown",
        "confidence_percent": confidence_percent,
        "confidence_label": confidence_label,
        "has_ocr_confidence": has_ocr_confidence,
        "ocr_confidence": ocr_confidence if has_ocr_confidence else 1.0,
        "meets_target": meets_target,
        "processing_mode": "summarize",
        "preview_kind": "pdf" if suffix == ".pdf" else "image",
        "preview_url": f"/preview/{upload_path.name}",
        "reviewed_text": extracted_text,
        "human_approved": False,
    }


def process_upload_job(
    job_id: str,
    upload_path: Path,
    filename: str,
    suffix: str,
    languages: list[str],
    language_label: str,
    direct_text: str = "",
    processing_mode: str = "summarize",
) -> None:
    try:
        if direct_text.strip():
            extracted_text = direct_text.strip()
            context = {
                "filename": "Direct text",
                "job_id": job_id,
                "language_label": "Not applicable",
                "page_count": 1,
                "summary": summarize_text(extracted_text),
                "ai_result": (
                    "Preparing the AI summary..."
                    if processing_mode == "summarize"
                    else "AI summary skipped because Keep Text As-Is was selected."
                ),
                "model_status": "processing" if processing_mode == "summarize" else "skipped",
                "model_input_truncated": False,
                "raw_text": extracted_text,
                "ocr_warning": None,
                "ocr_status": "direct_text",
                "pages": [],
                "backend": "direct_text",
                "confidence_percent": 100,
                "confidence_label": "Direct text — OCR not required",
                "has_ocr_confidence": True,
                "ocr_confidence": 1.0,
                "meets_target": True,
                "processing_mode": processing_mode,
                "preview_kind": "text",
                "preview_url": None,
                "reviewed_text": extracted_text,
                "human_approved": False,
            }
            update_job(
                job_id,
                stage="complete",
                progress=100,
                message="Direct text received. Ready for human review.",
                result=context,
                updated_at=time.time(),
            )
            if processing_mode == "summarize":
                worker = threading.Thread(
                    target=run_vlm_summary,
                    args=(job_id, extracted_text, 1.0),
                    daemon=True,
                )
                with PROCESSING_JOBS_LOCK:
                    PROCESSING_JOBS[job_id]["model_worker"] = worker
                worker.start()
            return

        update_job(
            job_id,
            stage="ocr",
            progress=15,
            message="File saved. Checking embedded text and pages that require OCR...",
        )
        with OCR_PROCESS_LOCK:
            update_job(
                job_id,
                stage="ocr",
                progress=20,
                message="Extracting embedded text or reading scanned pages...",
            )
            result = run_ocr_isolated(upload_path, languages)
        update_job(
            job_id,
            stage="model",
            progress=72,
            message="OCR complete. Evaluating confidence and preparing the result...",
        )
        context = build_result_context(
            result, filename, suffix, upload_path, job_id, language_label
        )
        context["processing_mode"] = processing_mode
        summary_request = None
        if (
            processing_mode == "summarize"
            and context["meets_target"]
            and result.get("ocr_status") != "no_text_detected"
        ):
            update_job(
                job_id,
                stage="model",
                progress=82,
                message="OCR complete. Preparing the AI summary...",
            )
            context["model_status"] = "processing"
            summary_request = (
                context["raw_text"],
                context["ocr_confidence"],
            )
        elif processing_mode == "as_is":
            context["model_status"] = "skipped"
            context["ai_result"] = "AI summary skipped because Keep Text As-Is was selected."
            update_job(
                job_id,
                stage="review",
                progress=90,
                message="Processing complete. Proceeding to human review.",
            )
        elif result.get("ocr_status") == "no_text_detected":
            update_job(
                job_id,
                stage="review",
                progress=90,
                message="No readable text was detected. Review the OCR diagnostic.",
            )
        else:
            update_job(
                job_id,
                stage="review",
                progress=90,
                message="Confidence is below 90%; the model summary was skipped.",
            )

        model_worker = None
        with PROCESSING_JOBS_LOCK:
            job = PROCESSING_JOBS.get(job_id)
            if job is not None:
                if summary_request is not None:
                    model_worker = threading.Thread(
                        target=run_vlm_summary,
                        args=(job_id, *summary_request),
                        daemon=True,
                    )
                job.update(
                    stage="complete",
                    progress=100,
                    message="Text extraction complete. Review the result; the AI summary will appear when complete."
                    if summary_request is not None
                    else "Processing complete. Review the extracted text before approval.",
                    result=context,
                    updated_at=time.time(),
                )
                if model_worker is not None:
                    job["model_worker"] = model_worker
        if model_worker is not None:
            model_worker.start()
    except Exception as exc:
        app.logger.exception("Document processing job %s failed", job_id)
        update_job(job_id, stage="failed", error=str(exc), message="Unable to complete processing.")
        if upload_path.exists():
            try:
                upload_path.unlink()
            except OSError:
                app.logger.warning("Could not remove failed upload: %s", upload_path)


def run_vlm_summary(job_id: str, text: str, ocr_confidence: float) -> None:
    response = {}
    try:
        response = call_vlm_service(
            text,
            task="summary",
            ocr_confidence=ocr_confidence,
        )
        model_status = "complete" if response.get("status") == "ok" else "error"
        ai_result = response.get("result") or "The model returned an empty summary."
        if model_status == "error":
            app.logger.warning("VLM summary failed for job %s: %s", job_id, ai_result)
            ai_result = "Local AI service (Ollama) is offline — showing extracted text only."
            model_status = "offline"
    except Exception as exc:
        app.logger.exception("VLM summary job %s failed", job_id)
        model_status = "offline"
        ai_result = "Local AI service (Ollama) is offline — showing extracted text only."

    with PROCESSING_JOBS_LOCK:
        job = PROCESSING_JOBS.get(job_id)
        if job is not None and job.get("result") is not None:
            job["result"]["model_status"] = model_status
            job["result"]["ai_result"] = ai_result
            job["result"]["model_input_truncated"] = bool(
                response.get("input_truncated", False)
            )
            job["updated_at"] = time.time()


def cleanup_expired_jobs() -> None:
    cutoff = time.time() - JOB_RETENTION_SECONDS
    with PROCESSING_JOBS_LOCK:
        expired = [
            job_id for job_id, job in PROCESSING_JOBS.items()
            if job.get("stage") in {"complete", "failed"}
            and job.get("updated_at", 0) < cutoff
        ]
        for job_id in expired:
            PROCESSING_JOBS.pop(job_id, None)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return {
        "status": "ok",
        "message": "CNC Smart Writer is running",
        "version": APP_BUILD_ID,
    }


@app.route("/preview/<filename>")
def preview_file(filename: str):
    if not UPLOAD_NAME_PATTERN.fullmatch(filename):
        abort(404)
    upload_dir = Path(app.config["UPLOAD_FOLDER"])
    if not (upload_dir / filename).is_file():
        abort(404)
    return send_from_directory(
        upload_dir,
        filename,
        mimetype="application/pdf" if filename.lower().endswith(".pdf") else None,
        as_attachment=False,
        conditional=True,
    )


@app.route("/process", methods=["POST"])
def process_document():
    file = request.files.get("document")
    direct_text = request.form.get("direct_text", "").strip()
    has_file = file is not None and file.filename != ""

    if has_file and direct_text:
        return render_template(
            "index.html",
            error="Choose either a file or direct text, not both.",
        ), 400

    if not has_file and not direct_text:
        return render_template("index.html", error="Choose a file or enter text to analyze."), 400

    if has_file and not allowed_file(file.filename):
        return render_template("index.html", error="Allowed types: PDF, JPG, PNG, BMP, TIFF, and WEBP."), 400

    languages, language_label = ["en"], "English (locked)"

    processing_mode = request.form.get("processing_mode", "summarize")
    if processing_mode not in {"summarize", "as_is"}:
        return render_template("index.html", error="Choose Summarize with AI or Keep Text As-Is."), 400

    upload_dir = app.config["UPLOAD_FOLDER"]
    upload_dir.mkdir(parents=True, exist_ok=True)
    cleanup_expired_uploads(upload_dir)

    if has_file:
        suffix = Path(file.filename).suffix.lower()
        temp_path = upload_dir / f"upload_{uuid.uuid4().hex}{suffix}"
        file.save(temp_path)
        original_filename = file.filename
    else:
        suffix = ".txt"
        temp_path = upload_dir / f"upload_{uuid.uuid4().hex}.txt"
        original_filename = "Direct text"

    job_id = uuid.uuid4().hex
    cleanup_expired_jobs()
    with PROCESSING_JOBS_LOCK:
        PROCESSING_JOBS[job_id] = {
            "stage": "queued",
            "progress": 5,
            "message": "Input received. Starting processing...",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        worker = threading.Thread(
            target=process_upload_job,
            args=(
                job_id,
                temp_path,
                original_filename,
                suffix,
                languages,
                language_label,
                direct_text,
                processing_mode,
            ),
            daemon=True,
        )
        PROCESSING_JOBS[job_id]["worker"] = worker
    worker.start()
    return render_template("processing.html", job_id=job_id)


@app.route("/status/<job_id>")
def processing_status(job_id: str):
    with PROCESSING_JOBS_LOCK:
        job = PROCESSING_JOBS.get(job_id)
        if job is None:
            abort(404)
        response = {
            "stage": job["stage"],
            "progress": job["progress"],
            "message": job["message"],
        }
        if job.get("result") is not None:
            response["model_status"] = job["result"]["model_status"]
            response["ai_result"] = job["result"]["ai_result"]
            response["model_input_truncated"] = job["result"][
                "model_input_truncated"
            ]
        if job["stage"] == "failed":
            response["error"] = job.get("error", "Processing failed.")
        if job["stage"] == "complete":
            response["result_url"] = f"/result/{job_id}"
    return response


@app.route("/result/<job_id>")
def completed_result(job_id: str):
    with PROCESSING_JOBS_LOCK:
        job = PROCESSING_JOBS.get(job_id)
        if job is None:
            abort(404)
        if job["stage"] == "failed":
            error = job.get("error", "Processing failed.")
        elif job["stage"] != "complete":
            return render_template("processing.html", job_id=job_id), 202
        else:
            if job["result"].get("human_approved"):
                job["result"]["gcode_preview"] = generate_starter_gcode(
                    job["result"]["reviewed_text"]
                )
            return render_template("result.html", **job["result"])
    return render_template("index.html", error=f"Processing failed: {error}"), 500


@app.route("/review/<job_id>", methods=["POST"])
def approve_review(job_id: str):
    reviewed_text = request.form.get("reviewed_text", "").strip()
    if not reviewed_text:
        abort(400, description="Reviewed text cannot be empty.")
    if request.form.get("approve") != "yes":
        abort(400, description="Explicit review approval is required.")

    with PROCESSING_JOBS_LOCK:
        job = PROCESSING_JOBS.get(job_id)
        if job is None or job.get("stage") != "complete":
            abort(404)
        job["result"]["reviewed_text"] = reviewed_text
        job["result"]["human_approved"] = True
        job["result"]["gcode_preview"] = generate_starter_gcode(reviewed_text)
        job["updated_at"] = time.time()
    return render_template("result.html", **job["result"])


@app.route("/download/<job_id>/txt")
def download_approved_text(job_id: str):
    with PROCESSING_JOBS_LOCK:
        job = PROCESSING_JOBS.get(job_id)
        if job is None or not job["result"].get("human_approved"):
            abort(404)
        text = job["result"]["reviewed_text"]

    data = BytesIO(text.encode("utf-8"))
    data.seek(0)
    return send_file(
        data,
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"{job_id}_approved.txt",
    )


@app.route("/download/<job_id>/gcode")
def download_approved_gcode(job_id: str):
    with PROCESSING_JOBS_LOCK:
        job = PROCESSING_JOBS.get(job_id)
        if job is None or not job["result"].get("human_approved"):
            abort(404)
        text = job["result"]["reviewed_text"]

    data = BytesIO(generate_starter_gcode(text).encode("utf-8"))
    data.seek(0)
    return send_file(
        data,
        mimetype="application/x-gcode",
        as_attachment=True,
        download_name=f"{job_id}_approved.gcode",
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False, threaded=True)
