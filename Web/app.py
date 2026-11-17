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

from flask import Flask, abort, render_template, request, send_from_directory
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


def run_ocr_isolated(source_path: Path, languages: list[str]) -> dict:
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
    has_arabic = any(
        "\u0600" <= char <= "\u06ff"
        or "\u0750" <= char <= "\u077f"
        or "\ufb50" <= char <= "\ufdff"
        or "\ufe70" <= char <= "\ufeff"
        for char in text
    )
    payload = json.dumps({
        "text": text,
        "task": task,
        "language": "ar" if has_arabic else "en",
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
            "تم إيقاف التلخيص الآلي لأن ثقة OCR أقل من 90%. راجع النص وصحح الأخطاء قبل اعتماد المستند."
            if any("\u0600" <= char <= "\u06ff" for char in extracted_text)
            else "AI summary was skipped because OCR confidence is below 90%. Review and correct the extracted text before approving it."
        )
        model_status = "skipped"
    else:
        ai_result = "اكتمل OCR. يجري إعداد ملخص النموذج في الخلفية..."
        model_status = "pending"

    detected_language = result.get("detected_language", "unknown")
    detected_labels = {
        "ar": "Arabic (auto-detected)",
        "en": "English (auto-detected)",
        "mixed": "Arabic + English (auto-detected)",
    }
    if language_label == "Auto-detect":
        language_label = detected_labels.get(
            detected_language, "Auto-detect (scanned pages use OCR)"
        )

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
) -> None:
    try:
        update_job(
            job_id,
            stage="ocr",
            progress=15,
            message="تم حفظ الملف. جاري فحص النص الرقمي وتحديد الصفحات التي تحتاج OCR...",
        )
        with OCR_PROCESS_LOCK:
            update_job(
                job_id,
                stage="ocr",
                progress=20,
                message="جاري استخراج النص الرقمي أو قراءة الصفحات المصورة وترتيب النص...",
            )
            result = run_ocr_isolated(upload_path, languages)
        update_job(
            job_id,
            stage="model",
            progress=72,
            message="اكتمل OCR. جاري تقييم الثقة وتجهيز النتيجة...",
        )
        context = build_result_context(
            result, filename, suffix, upload_path, job_id, language_label
        )
        summary_request = None
        if context["meets_target"] and result.get("ocr_status") != "no_text_detected":
            update_job(
                job_id,
                stage="model",
                progress=82,
                message="اكتمل OCR. جاري تجهيز النتيجة؛ سيظهر ملخص النموذج عند اكتماله.",
            )
            context["model_status"] = "processing"
            summary_request = (
                context["raw_text"],
                context["ocr_confidence"],
            )
        elif result.get("ocr_status") == "no_text_detected":
            update_job(
                job_id,
                stage="review",
                progress=90,
                message="لم يتم اكتشاف نص مقروء. راجع تشخيص OCR والصورة الأصلية.",
            )
        else:
            update_job(
                job_id,
                stage="review",
                progress=90,
                message="الثقة أقل من 90%؛ تم تخطي النموذج لتجنب تلخيص قراءة غير مؤكدة.",
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
                    message="اكتمل استخراج النص. راجعه الآن؛ ملخص النموذج سيظهر عند اكتماله."
                    if summary_request is not None
                    else "اكتملت المعالجة. راجع النص الأصلي والمستخرج قبل الاعتماد.",
                    result=context,
                    updated_at=time.time(),
                )
                if model_worker is not None:
                    job["model_worker"] = model_worker
        if model_worker is not None:
            model_worker.start()
    except Exception as exc:
        app.logger.exception("Document processing job %s failed", job_id)
        update_job(job_id, stage="failed", error=str(exc), message="تعذر إكمال المعالجة.")
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
    except Exception as exc:
        app.logger.exception("VLM summary job %s failed", job_id)
        model_status = "error"
        ai_result = f"VLM summary failed: {exc}"

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
        "message": "OCR web demo is running",
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
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose a file first."), 400

    if not allowed_file(file.filename):
        return render_template("index.html", error="Allowed types: PDF, JPG, PNG, BMP, TIFF, WEBP."), 400

    language = request.form.get("language", "auto")
    language_options = {
        "ar": (["ar"], "Arabic"),
        "en": (["en"], "English"),
        "auto": (["auto"], "Auto-detect"),
        "mixed": (["auto"], "Auto-detect"),
    }
    if language not in language_options:
        return render_template("index.html", error="Choose automatic, Arabic, or English language detection."), 400
    languages, language_label = language_options[language]

    upload_dir = app.config["UPLOAD_FOLDER"]
    upload_dir.mkdir(parents=True, exist_ok=True)
    cleanup_expired_uploads(upload_dir)
    suffix = Path(file.filename).suffix.lower()
    temp_path = upload_dir / f"upload_{uuid.uuid4().hex}{suffix}"
    file.save(temp_path)
    job_id = uuid.uuid4().hex
    cleanup_expired_jobs()
    with PROCESSING_JOBS_LOCK:
        PROCESSING_JOBS[job_id] = {
            "stage": "queued",
            "progress": 5,
            "message": "تم استلام الملف. جاري بدء المعالجة...",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        worker = threading.Thread(
            target=process_upload_job,
            args=(job_id, temp_path, file.filename, suffix, languages, language_label),
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
        job["updated_at"] = time.time()
    return render_template("result.html", **job["result"])


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False, threaded=True)
