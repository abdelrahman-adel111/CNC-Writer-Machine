import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
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
MAX_FILE_BYTES = 16 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_BYTES + 1024 * 1024
app.config["UPLOAD_FOLDER"] = PROJECT_ROOT / "Web" / "uploads"
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}
TEXT_DOCUMENT_EXTENSIONS = {".docx", ".txt"}
UPLOAD_RETENTION_SECONDS = 24 * 60 * 60
UPLOAD_NAME_PATTERN = re.compile(r"upload_[0-9a-f]{32}\.(?:png|jpe?g|pdf|docx|txt)\Z")
UPLOAD_CLEANUP_PATTERN = re.compile(r"upload_[a-zA-Z0-9_-]+\.(?:png|jpe?g|pdf|docx|txt|bmp|tiff?|webp)\Z")
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


def extract_text_document(source_path: Path) -> str:
    if source_path.suffix.lower() == ".txt":
        try:
            text = source_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("The TXT file must use UTF-8 encoding.") from exc
    elif source_path.suffix.lower() == ".docx":
        try:
            with zipfile.ZipFile(source_path) as document:
                document_xml = document.getinfo("word/document.xml")
                if document_xml.file_size > 32 * 1024 * 1024:
                    raise ValueError("The DOCX document contains too much text to process.")
                root = ET.fromstring(document.read(document_xml))
        except (KeyError, zipfile.BadZipFile, ET.ParseError) as exc:
            raise ValueError("The DOCX file is invalid or could not be read.") from exc

        word_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs = []
        for paragraph in root.iter(f"{word_ns}p"):
            pieces = []
            for element in paragraph.iter():
                if element.tag == f"{word_ns}t":
                    pieces.append(element.text or "")
                elif element.tag == f"{word_ns}tab":
                    pieces.append("\t")
                elif element.tag == f"{word_ns}br":
                    pieces.append("\n")
            paragraphs.append("".join(pieces))
        text = "\n".join(paragraphs)
    else:
        raise ValueError("Only DOCX and TXT documents can be read as text files.")

    text = text.strip()
    if not text:
        raise ValueError("No readable text was found in the uploaded file.")
    return text


def build_pipeline_steps(input_kind: str, processing_mode: str) -> list[dict]:
    steps = []
    if input_kind == "file":
        steps.append({"key": "extract", "label": "Extract Text"})
    if processing_mode == "summarize":
        steps.append({"key": "summarize", "label": "AI Summarization"})
    steps.append({"key": "gcode", "label": "Generate G-Code"})
    return steps


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
        "; CNC Smart Writer — Starter Pen Plotter Program",
        "; Human-verified text preparation (motion placeholders)",
        "G21 ; Set units to millimeters",
        "G90 ; Absolute positioning",
        "G00 Z5.0 ; Pen UP (safe clearance)",
        "",
    ]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line_number, line in enumerate(lines, start=1):
        y = 24 + (line_number - 1) * 16
        x_end = min(195, 14 + len(line) * 6.2)
        gcode_lines.extend([
            f"; Line {line_number}: {line}",
            f"G00 X14.0 Y{y:.1f}",
            "G01 Z0.0 F600 ; Pen DOWN",
            f"G01 X{x_end:.1f} Y{y:.1f} F1200",
            "G00 Z5.0 ; Pen UP",
            "",
        ])
    gcode_lines.extend(["G00 X0.0 Y0.0 ; Return to origin", "M30 ; End program"])
    return "\n".join(gcode_lines)


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

    if suffix in TEXT_DOCUMENT_EXTENSIONS:
        confidence_label = "Text extracted directly from the uploaded file"
    elif not has_ocr_confidence and any(page.get("backend") == "pdf_text" for page in pages):
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
        "preview_kind": (
            "text" if suffix in TEXT_DOCUMENT_EXTENSIONS
            else ("pdf" if suffix == ".pdf" else "image")
        ),
        "preview_url": (
            None if suffix in TEXT_DOCUMENT_EXTENSIONS
            else f"/preview/{upload_path.name}"
        ),
        "reviewed_text": extracted_text,
        "human_approved": False,
        "gcode_preview": generate_starter_gcode(extracted_text),
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
            input_kind = "direct_text"
            suffix = ".txt"
            result = {
                "text": direct_text.strip(),
                "pages": [{
                    "page": 1,
                    "backend": "direct_text",
                    "confidence": 1.0,
                    "confidence_source": "direct_text",
                    "text": direct_text.strip(),
                }],
                "page_count": 1,
                "ocr_status": "direct_text",
                "warning": None,
            }
        else:
            input_kind = "file"
            update_job(
                job_id,
                stage="ocr",
                progress=12,
                message="Extracting text from the uploaded file...",
            )
            if suffix in TEXT_DOCUMENT_EXTENSIONS:
                extracted_text = extract_text_document(upload_path)
                result = {
                    "text": extracted_text,
                    "pages": [{
                        "page": 1,
                        "backend": "file_text",
                        "confidence": 1.0,
                        "confidence_source": "file_text",
                        "text": extracted_text,
                    }],
                    "page_count": 1,
                    "ocr_status": "file_text",
                    "warning": None,
                }
            else:
                with OCR_PROCESS_LOCK:
                    update_job(
                        job_id,
                        stage="ocr",
                        progress=20,
                        message="Reading document pages and extracting text...",
                    )
                    result = run_ocr_isolated(upload_path, languages)

        context = build_result_context(
            result, filename, suffix, upload_path, job_id, language_label
        )
        context["processing_mode"] = processing_mode
        context["input_kind"] = input_kind
        if input_kind == "direct_text":
            context["preview_url"] = None
        completed_steps = ["extract"] if input_kind == "file" else []

        if (
            processing_mode == "summarize"
            and context["meets_target"]
            and result.get("ocr_status") != "no_text_detected"
        ):
            context["model_status"] = "processing"
            context["ai_result"] = "AI summarization is in progress."
            update_job(
                job_id,
                stage="summarize",
                progress=60,
                message="Summarizing the extracted text with local AI...",
                current_step="summarize",
                completed_steps=completed_steps,
                result=context,
            )
            run_vlm_summary(job_id, context["raw_text"], context["ocr_confidence"])
            with PROCESSING_JOBS_LOCK:
                job = PROCESSING_JOBS.get(job_id)
                if job is not None and job.get("result") is not None:
                    context = job["result"]
            completed_steps.append("summarize")
        else:
            context["model_status"] = "skipped"
            if processing_mode == "as_is":
                context["ai_result"] = "Summarization was not selected."
            elif result.get("ocr_status") == "no_text_detected":
                context["ai_result"] = (
                    "AI summarization was skipped because no text was extracted."
                )
            else:
                context["ai_result"] = (
                    "AI summarization was skipped because text recognition confidence "
                    "was below 90%."
                )
            context["reviewed_text"] = context["raw_text"]
            if processing_mode == "summarize":
                completed_steps.append("summarize")

        update_job(
            job_id,
            stage="gcode",
            progress=88,
            message="Generating the G-Code preview...",
            current_step="gcode",
            completed_steps=completed_steps,
            result=context,
        )
        context["gcode_preview"] = generate_starter_gcode(context["reviewed_text"])
        update_job(
            job_id,
            stage="complete",
            progress=100,
            message="Processing complete. Continue to output for verification.",
            current_step=None,
            completed_steps=completed_steps + ["gcode"],
            result=context,
        )
    except Exception as exc:
        app.logger.exception("Document processing job %s failed", job_id)
        update_job(job_id, stage="failed", error=str(exc), message="Unable to complete processing.")


def run_vlm_summary(job_id: str, text: str, ocr_confidence: float) -> None:
    response = {}
    try:
        response = call_vlm_service(
            text,
            task="summary",
            ocr_confidence=ocr_confidence,
        )
        model_status = "complete" if response.get("status") == "ok" else "error"
        result_text = response.get("result")
        ai_result = (
            result_text.strip()
            if isinstance(result_text, str) and result_text.strip()
            else "The model returned an empty summary."
        )
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
            job["result"]["reviewed_text"] = (
                ai_result if model_status == "complete" else job["result"]["raw_text"]
            )
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


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return render_template(
        "index.html",
        error="The upload exceeds the 16 MB limit. Choose a smaller file.",
    ), 413


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
        return render_template(
            "index.html",
            error="Allowed file types: PDF, DOCX, TXT, PNG, JPG, and JPEG.",
        ), 400

    if has_file:
        file.stream.seek(0, os.SEEK_END)
        file_size = file.stream.tell()
        file.stream.seek(0)
        if file_size == 0:
            return render_template(
                "index.html",
                error="The selected file is empty. Choose a file containing data.",
            ), 400
        if file_size > MAX_FILE_BYTES:
            return render_template(
                "index.html",
                error="The upload exceeds the 16 MB limit. Choose a smaller file.",
            ), 413

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
    input_kind = "file" if has_file else "direct_text"
    pipeline_steps = build_pipeline_steps(input_kind, processing_mode)
    worker_args = (
        job_id,
        temp_path,
        original_filename,
        suffix,
        languages,
        language_label,
        direct_text,
        processing_mode,
    )
    with PROCESSING_JOBS_LOCK:
        PROCESSING_JOBS[job_id] = {
            "stage": "queued",
            "progress": 5,
            "message": "Input received. Starting processing...",
            "current_step": pipeline_steps[0]["key"],
            "completed_steps": [],
            "pipeline_steps": pipeline_steps,
            "processing_mode": processing_mode,
            "input_kind": input_kind,
            "preview_kind": (
                "text" if (not has_file or suffix in TEXT_DOCUMENT_EXTENSIONS)
                else ("pdf" if suffix == ".pdf" else "image")
            ),
            "preview_url": f"/preview/{temp_path.name}" if has_file else None,
            "preview_text": direct_text[:1600],
            "filename": original_filename,
            "retry_args": worker_args[1:],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        worker = threading.Thread(target=process_upload_job, args=worker_args, daemon=True)
        PROCESSING_JOBS[job_id]["worker"] = worker
    worker.start()
    return render_template(
        "processing.html",
        job_id=job_id,
        preview_kind=PROCESSING_JOBS[job_id]["preview_kind"],
        preview_url=PROCESSING_JOBS[job_id]["preview_url"],
        preview_text=PROCESSING_JOBS[job_id]["preview_text"],
        filename=PROCESSING_JOBS[job_id]["filename"],
        pipeline_steps=pipeline_steps,
    )


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
            "processing_mode": job.get("processing_mode", "summarize"),
            "input_kind": job.get("input_kind", "file"),
            "pipeline_steps": job.get("pipeline_steps", []),
            "completed_steps": job.get("completed_steps", []),
            "current_step": job.get("current_step"),
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
            return render_template(
                "processing.html",
                job_id=job_id,
                preview_kind=job.get("preview_kind", "text"),
                preview_url=job.get("preview_url"),
                preview_text=job.get("preview_text", ""),
                filename=job.get("filename", ""),
                pipeline_steps=job.get("pipeline_steps", []),
            ), 202
        else:
            if job["result"].get("human_approved"):
                job["result"]["gcode_preview"] = generate_starter_gcode(
                    job["result"]["reviewed_text"]
                )
            return render_template("result.html", **job["result"])
    return render_template("index.html", error=f"Processing failed: {error}"), 500


@app.route("/retry/<job_id>", methods=["POST"])
def retry_processing(job_id: str):
    with PROCESSING_JOBS_LOCK:
        job = PROCESSING_JOBS.get(job_id)
        if job is None or job.get("stage") != "failed":
            abort(404)
        retry_args = job.get("retry_args")
        if retry_args is None:
            abort(404)
        pipeline_steps = job.get("pipeline_steps", [])
        if not pipeline_steps:
            abort(500)
        job.update(
            stage="queued",
            progress=5,
            message="Retrying document processing...",
            current_step=pipeline_steps[0]["key"],
            completed_steps=[],
            result=None,
            error=None,
            updated_at=time.time(),
        )
        worker_args = (job_id, *retry_args)
        worker = threading.Thread(
            target=process_upload_job, args=worker_args, daemon=True
        )
        job["worker"] = worker
        preview = {
            "preview_kind": job.get("preview_kind", "text"),
            "preview_url": job.get("preview_url"),
            "preview_text": job.get("preview_text", ""),
            "filename": job.get("filename", ""),
        }
    worker.start()
    return render_template(
        "processing.html",
        job_id=job_id,
        pipeline_steps=pipeline_steps,
        **preview,
    )


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
