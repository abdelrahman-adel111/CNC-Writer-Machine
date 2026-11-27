import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, patch

import pymupdf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Web.app import (
    OCR_WORKER_TIMEOUT_SECONDS,
    OCR_SECONDS_PER_SCANNED_PAGE,
    PROCESSING_JOBS,
    PROCESSING_JOBS_LOCK,
    VLM_SERVICE_TIMEOUT_SECONDS,
    app,
    call_vlm_service,
    ocr_worker_timeout,
)


class ProcessingFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="ocr_flow_test_")
        self.upload_dir = Path(self.temp_dir.name)
        self.previous_upload_dir = app.config["UPLOAD_FOLDER"]
        app.config["UPLOAD_FOLDER"] = self.upload_dir
        self.client = app.test_client()
        self.release_events = []
        with PROCESSING_JOBS_LOCK:
            PROCESSING_JOBS.clear()

        image_buffer = io.BytesIO()
        Image.new("RGB", (20, 20), "white").save(image_buffer, format="PNG")
        self.image_bytes = image_buffer.getvalue()

    def tearDown(self):
        for event in self.release_events:
            event.set()
        with PROCESSING_JOBS_LOCK:
            workers = [
                worker
                for job in PROCESSING_JOBS.values()
                for worker in (job.get("worker"), job.get("model_worker"))
                if worker is not None
            ]
        for worker in workers:
            worker.join(timeout=5)
        app.config["UPLOAD_FOLDER"] = self.previous_upload_dir
        with PROCESSING_JOBS_LOCK:
            PROCESSING_JOBS.clear()
        self.client = None
        self.temp_dir.cleanup()

    def wait_for_status(self, job_id, stage, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.client.get(f"/status/{job_id}")
            self.assertEqual(response.status_code, 200)
            status = response.get_json()
            if status["stage"] == stage:
                return status
            time.sleep(0.01)
        self.fail(f"Job {job_id} did not reach stage {stage!r}.")

    def wait_for_model_status(self, job_id, status, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.client.get(f"/status/{job_id}")
            self.assertEqual(response.status_code, 200)
            state = response.get_json()
            if state.get("model_status") == status:
                return state
            time.sleep(0.01)
        self.fail(f"Job {job_id} did not reach model status {status!r}.")

    def test_real_job_stages_review_gate_and_previews(self):
        ocr_started = Event()
        continue_ocr = Event()
        model_started = Event()
        continue_model = Event()
        self.release_events.extend([continue_ocr, continue_model])
        ocr_result = {
            "status": "ok",
            "ocr_status": "ready",
            "page_count": 1,
            "text": "Recognized source text",
            "pages": [{
                "page": 1,
                "backend": "paddleocr",
                "confidence": 0.95,
                "text": "Recognized source text",
            }],
            "warning": None,
        }

        def fake_ocr(_path, languages):
            self.assertEqual(languages, ["en"])
            ocr_started.set()
            self.assertTrue(continue_ocr.wait(5))
            return ocr_result

        def fake_model(_text, task, ocr_confidence):
            self.assertEqual(task, "summary")
            self.assertEqual(ocr_confidence, 0.95)
            model_started.set()
            self.assertTrue(continue_model.wait(5))
            return {"status": "ok", "result": "Grounded summary"}

        with patch("Web.app.run_ocr_isolated", side_effect=fake_ocr), patch(
            "Web.app.call_vlm_service", side_effect=fake_model
        ):
            response = self.client.post(
                "/process",
                data={"document": (io.BytesIO(self.image_bytes), "sample.png")},
                content_type="multipart/form-data",
            )
            self.assertEqual(response.status_code, 200)
            job_id = response.get_data(as_text=True).split('const jobId = "')[1].split('"')[0]

            self.assertTrue(ocr_started.wait(5))
            self.assertEqual(self.client.get(f"/status/{job_id}").get_json()["stage"], "ocr")
            continue_ocr.set()

            self.assertTrue(model_started.wait(5))
            self.assertEqual(
                self.client.get(f"/status/{job_id}").get_json()["stage"],
                "summarize",
            )
            continue_model.set()
            completed = self.wait_for_status(job_id, "complete")
            self.assertEqual(completed.get("model_status"), "complete")
            result_response = self.client.get(completed["result_url"])
            self.assertEqual(result_response.status_code, 200)
            result_html = result_response.get_data(as_text=True)
            result_response.close()
            self.assertIn("Extracted Text", result_html)
            self.assertIn("Grounded summary", result_html)

            denied = self.client.post(
                f"/review/{job_id}",
                data={"reviewed_text": "unreviewed"},
            )
            self.assertEqual(denied.status_code, 400)
            approved = self.client.post(
                f"/review/{job_id}",
                data={"reviewed_text": "Human corrected text", "approve": "yes"},
            )
            self.assertEqual(approved.status_code, 200)
            approved_html = approved.get_data(as_text=True)
            self.assertIn("Approve &amp; Generate G-Code", approved_html)

            with PROCESSING_JOBS_LOCK:
                upload_name = PROCESSING_JOBS[job_id]["result"]["preview_url"].split("/")[-1]
            preview = self.client.get(f"/preview/{upload_name}")
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview.data, self.image_bytes)
            preview.close()

    def test_low_confidence_skips_model_and_keeps_human_review(self):
        result = {
            "status": "ok",
            "ocr_status": "ready",
            "page_count": 1,
            "text": "uncertain text",
            "pages": [{
                "page": 1,
                "backend": "paddleocr",
                "confidence": 0.89,
                "text": "uncertain text",
            }],
            "warning": None,
        }
        with patch("Web.app.run_ocr_isolated", return_value=result), patch(
            "Web.app.call_vlm_service"
        ) as model:
            response = self.client.post(
                "/process",
                data={"document": (io.BytesIO(self.image_bytes), "sample.png")},
                content_type="multipart/form-data",
            )
            job_id = response.get_data(as_text=True).split('const jobId = "')[1].split('"')[0]
            status = self.wait_for_status(job_id, "complete")
            self.assertEqual(response.status_code, 200)
            self.assertFalse(model.called)
            page = self.client.get(status["result_url"])
            html = page.get_data(as_text=True)
            self.assertIn("AI Summary", html)
            self.assertIn("text recognition confidence was below 90%", html)
            page.close()

    def test_vlm_failure_does_not_hide_completed_ocr_result(self):
        result = {
            "status": "ok",
            "ocr_status": "ready",
            "page_count": 1,
            "text": "Recognized source text",
            "pages": [{
                "page": 1,
                "backend": "paddleocr",
                "confidence": 0.95,
                "text": "Recognized source text",
            }],
            "warning": None,
        }
        with patch("Web.app.run_ocr_isolated", return_value=result), patch(
            "Web.app.call_vlm_service",
            side_effect=RuntimeError("Ollama timed out"),
        ):
            response = self.client.post(
                "/process",
                data={"document": (io.BytesIO(self.image_bytes), "sample.png")},
                content_type="multipart/form-data",
            )
            job_id = response.get_data(as_text=True).split('const jobId = "')[1].split('"')[0]
            completed = self.wait_for_status(job_id, "complete")
            model_result = self.wait_for_model_status(job_id, "offline")

        self.assertEqual(completed["stage"], "complete")
        self.assertEqual(model_result["model_status"], "offline")
        self.assertEqual(
            model_result["ai_result"],
            "Local AI service (Ollama) is offline — showing extracted text only.",
        )
        self.assertFalse(model_result["model_input_truncated"])
        result_page = self.client.get(completed["result_url"])
        self.assertIn("Recognized source text", result_page.get_data(as_text=True))
        result_page.close()

    def test_long_document_summary_discloses_that_excerpts_were_used(self):
        long_text = "[Page 1]\n" + ("lecture content " * 400)
        ocr_result = {
            "status": "ok",
            "ocr_status": "ready",
            "page_count": 1,
            "text": long_text,
            "pages": [{
                "page": 1,
                "backend": "pdf_text",
                "confidence": 0.0,
                "confidence_source": "pdf_text",
                "text": long_text,
            }],
            "warning": None,
        }
        with patch("Web.app.run_ocr_isolated", return_value=ocr_result), patch(
            "Web.app.call_vlm_service",
            return_value={
                "status": "ok",
                "result": "Lecture summary",
                "input_truncated": True,
            },
        ):
            response = self.client.post(
                "/process",
                data={"document": (io.BytesIO(self.image_bytes), "lecture.pdf")},
                content_type="multipart/form-data",
            )
            job_id = response.get_data(as_text=True).split('const jobId = "')[1].split('"')[0]
            completed = self.wait_for_status(job_id, "complete")
            completed = self.wait_for_model_status(job_id, "complete")
            result_page = self.client.get(completed["result_url"])
            html = result_page.get_data(as_text=True)

        self.assertIn("Lecture summary", html)
        self.assertIn("AI Summary", html)
        self.assertIn("lecture content", html)
        result_page.close()

    def test_pdf_upload_uses_english_ocr_and_keeps_pdf_preview(self):
        document = pymupdf.open()
        document.new_page().insert_text((72, 72), "PDF OCR page")
        pdf_bytes = document.tobytes()
        document.close()
        result = {
            "status": "ok",
            "ocr_status": "ready",
            "page_count": 1,
            "text": "PDF OCR page",
            "detected_language": "en",
            "pages": [{
                "page": 1,
                "backend": "pdf_text",
                "confidence": 0,
                "confidence_source": "pdf_text",
                "text": "PDF OCR page",
            }],
            "warning": None,
        }

        with patch("Web.app.run_ocr_isolated", return_value=result) as ocr, patch(
            "Web.app.call_vlm_service",
            return_value={"status": "ok", "result": "PDF summary"},
        ):
            response = self.client.post(
                "/process",
                data={
                    "document": (io.BytesIO(pdf_bytes), "lecture.pdf"),
                },
                content_type="multipart/form-data",
            )
            job_id = response.get_data(as_text=True).split('const jobId = "')[1].split('"')[0]
            completed = self.wait_for_status(job_id, "complete")
            completed = self.wait_for_model_status(job_id, "complete")
            page = self.client.get(completed["result_url"])
            self.assertEqual(ocr.call_args.args[1], ["en"])
            self.assertEqual(Path(ocr.call_args.args[0]).suffix, ".pdf")
            self.assertIn("PDF summary", page.get_data(as_text=True))
            self.assertIn("AI Summary", page.get_data(as_text=True))
            page.close()

            with PROCESSING_JOBS_LOCK:
                upload_name = PROCESSING_JOBS[job_id]["result"]["preview_url"].split("/")[-1]
            preview = self.client.get(f"/preview/{upload_name}")
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview.mimetype, "application/pdf")
            self.assertEqual(preview.data, pdf_bytes)
            preview.close()

    def test_searchable_pdf_upload_uses_direct_text_extraction(self):
        document = pymupdf.open()
        page = document.new_page(width=612, height=792)
        page.insert_text(
            (72, 72),
            "A searchable English lecture page extracted directly from its PDF text layer.",
        )
        pdf_bytes = document.tobytes()
        document.close()

        with patch(
            "Web.app.call_vlm_service",
            return_value={"status": "ok", "result": "Direct PDF summary"},
        ):
            response = self.client.post(
                "/process",
                data={"document": (io.BytesIO(pdf_bytes), "lecture.pdf")},
                content_type="multipart/form-data",
            )
            job_id = response.get_data(as_text=True).split('const jobId = "')[1].split('"')[0]
            completed = self.wait_for_status(job_id, "complete", timeout=60)
            completed = self.wait_for_model_status(job_id, "complete", timeout=60)
            page = self.client.get(completed["result_url"])
            html = page.get_data(as_text=True)
            self.assertIn("Direct PDF summary", html)
            self.assertIn("Extracted Text", html)
            self.assertIn(
                "A searchable English lecture page extracted directly from its PDF text layer.",
                html,
            )
            with PROCESSING_JOBS_LOCK:
                self.assertEqual(
                    PROCESSING_JOBS[job_id]["result"]["backend"], "pdf_text"
                )
            page.close()

    def test_scanned_pdf_timeout_scales_with_ocr_page_count(self):
        document = pymupdf.open()
        document.new_page().insert_text(
            (72, 72), "This searchable page has enough text to avoid OCR."
        )
        document.new_page(width=72, height=72)
        pdf_bytes = document.tobytes()
        document.close()
        pdf_path = self.upload_dir / "scanned.pdf"
        pdf_path.write_bytes(pdf_bytes)

        self.assertEqual(
            ocr_worker_timeout(pdf_path),
            OCR_WORKER_TIMEOUT_SECONDS + OCR_SECONDS_PER_SCANNED_PAGE,
        )

    def test_health_reports_current_build_for_launcher_staleness_check(self):
        health = self.client.get("/health").get_json()
        self.assertEqual(health["version"], "auto-pdf-text-v7")

    def test_vlm_http_request_timeout_is_bounded(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"status":"ok","result":"summary"}'
        with patch("Web.app.urllib.request.urlopen", return_value=response) as urlopen:
            result = call_vlm_service("Document text")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            urlopen.call_args.kwargs["timeout"], VLM_SERVICE_TIMEOUT_SECONDS
        )
        self.assertLessEqual(VLM_SERVICE_TIMEOUT_SECONDS, 100)
        self.assertGreater(VLM_SERVICE_TIMEOUT_SECONDS, 90)

    def test_upload_form_is_locked_to_english_without_language_selector(self):
        response = self.client.get("/")
        page = response.get_data(as_text=True)
        self.assertIn('<input type="hidden" name="language" value="en">', page)
        self.assertNotIn("Document language", page)

    def test_english_selection_uses_the_english_recognition_model(self):
        result = {
            "status": "ok",
            "ocr_status": "ready",
            "page_count": 1,
            "text": "Hello OCR Test",
            "pages": [{
                "page": 1,
                "backend": "paddleocr",
                "confidence": 0.99,
                "text": "Hello OCR Test",
            }],
            "warning": None,
        }
        with patch("Web.app.run_ocr_isolated", return_value=result) as ocr, patch(
            "Web.app.call_vlm_service",
            return_value={"status": "ok", "result": "English summary"},
        ):
            response = self.client.post(
                "/process",
                data={
                    "language": "en",
                    "document": (io.BytesIO(self.image_bytes), "مستند انجليزي.png"),
                },
                content_type="multipart/form-data",
            )
            job_id = response.get_data(as_text=True).split('const jobId = "')[1].split('"')[0]
            completed = self.wait_for_status(job_id, "complete")
            completed = self.wait_for_model_status(job_id, "complete")
            page = self.client.get(completed["result_url"])
            self.assertEqual(ocr.call_args.args[1], ["en"])
            self.assertEqual(Path(ocr.call_args.args[0]).suffix, ".png")
            self.assertIn("English", page.get_data(as_text=True))
            page.close()


if __name__ == "__main__":
    unittest.main()
