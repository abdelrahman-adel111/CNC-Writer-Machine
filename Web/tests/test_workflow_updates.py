import io
import json
import re
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Web.app import (
    MAX_FILE_BYTES,
    PROCESSING_JOBS,
    PROCESSING_JOBS_LOCK,
    app,
    build_pipeline_steps,
    call_vlm_service,
    extract_text_document,
)


class WorkflowUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="cnc_workflow_test_")
        self.previous_upload_dir = app.config["UPLOAD_FOLDER"]
        app.config["UPLOAD_FOLDER"] = Path(self.temp_dir.name)
        self.client = app.test_client()
        with PROCESSING_JOBS_LOCK:
            PROCESSING_JOBS.clear()

    def tearDown(self):
        with PROCESSING_JOBS_LOCK:
            workers = [
                job.get("worker")
                for job in PROCESSING_JOBS.values()
                if job.get("worker") is not None
            ]
        for worker in workers:
            worker.join(timeout=5)
        with PROCESSING_JOBS_LOCK:
            PROCESSING_JOBS.clear()
        app.config["UPLOAD_FOLDER"] = self.previous_upload_dir
        self.temp_dir.cleanup()

    def wait_for_status(self, job_id, stage, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.client.get(f"/status/{job_id}").get_json()
            if status["stage"] == stage:
                return status
            time.sleep(0.01)
        self.fail(f"Job {job_id} did not reach stage {stage!r}.")

    def job_id_from_response(self, response):
        match = re.search(
            r'const jobId = "([a-f0-9]+)"', response.get_data(as_text=True)
        )
        self.assertIsNotNone(match)
        return match.group(1)

    def test_pipeline_steps_follow_input_and_processing_mode(self):
        self.assertEqual(
            [step["label"] for step in build_pipeline_steps("file", "summarize")],
            ["Extract Text", "AI Summarization", "Generate G-Code"],
        )
        self.assertEqual(
            [step["label"] for step in build_pipeline_steps("file", "as_is")],
            ["Extract Text", "Generate G-Code"],
        )
        self.assertEqual(
            [step["label"] for step in build_pipeline_steps("direct_text", "summarize")],
            ["AI Summarization", "Generate G-Code"],
        )
        self.assertEqual(
            [step["label"] for step in build_pipeline_steps("direct_text", "as_is")],
            ["Generate G-Code"],
        )

    def test_direct_text_as_is_skips_ocr_and_approval_gates_downloads(self):
        with patch("Web.app.run_ocr_isolated") as ocr, patch(
            "Web.app.call_vlm_service"
        ) as model:
            response = self.client.post(
                "/process",
                data={"direct_text": "  Verify this line.  ", "processing_mode": "as_is"},
            )
            self.assertEqual(response.status_code, 200)
            job_id = self.job_id_from_response(response)
            status = self.wait_for_status(job_id, "complete")

        self.assertEqual(status["input_kind"], "direct_text")
        self.assertEqual([step["key"] for step in status["pipeline_steps"]], ["gcode"])
        self.assertEqual(status["completed_steps"], ["gcode"])
        ocr.assert_not_called()
        model.assert_not_called()

        result = self.client.get(status["result_url"])
        self.assertIn("Source Text", result.get_data(as_text=True))
        self.assertEqual(self.client.get(f"/download/{job_id}/txt").status_code, 404)
        approved = self.client.post(
            f"/review/{job_id}",
            data={"reviewed_text": "Verified output", "approve": "yes"},
        )
        self.assertEqual(approved.status_code, 200)
        download = self.client.get(f"/download/{job_id}/txt")
        self.assertEqual(download.data.decode("utf-8"), "Verified output")

    def test_txt_file_uses_extraction_step_and_offline_summary_fallback(self):
        with patch(
            "Web.app.call_vlm_service",
            return_value={"status": "error", "result": "Connection refused"},
        ) as model:
            response = self.client.post(
                "/process",
                data={
                    "document": (io.BytesIO(b"Uploaded text source"), "source.txt"),
                    "processing_mode": "summarize",
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(response.status_code, 200)
            job_id = self.job_id_from_response(response)
            status = self.wait_for_status(job_id, "complete")

        self.assertEqual(
            [step["key"] for step in status["pipeline_steps"]],
            ["extract", "summarize", "gcode"],
        )
        self.assertEqual(status["completed_steps"], ["extract", "summarize", "gcode"])
        self.assertEqual(status["model_status"], "offline")
        model.assert_called_once()
        result = self.client.get(status["result_url"]).get_data(as_text=True)
        self.assertIn("Extracted Text", result)
        self.assertIn(
            "Local AI service (Ollama) is offline — showing extracted text only.",
            result,
        )
        self.assertIn("Uploaded text source", result)

    def test_docx_text_extraction_reads_paragraphs_and_breaks(self):
        docx_path = Path(self.temp_dir.name) / "document.docx"
        document_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>First line</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Second line</w:t></w:r></w:p></w:body></w:document>"
        )
        with zipfile.ZipFile(docx_path, "w") as document:
            document.writestr("word/document.xml", document_xml)

        self.assertEqual(extract_text_document(docx_path), "First line\nSecond line")

    def test_upload_validation_rejects_unsupported_and_oversized_files(self):
        unsupported = self.client.post(
            "/process",
            data={"document": (io.BytesIO(b"data"), "source.bmp")},
            content_type="multipart/form-data",
        )
        self.assertEqual(unsupported.status_code, 400)
        self.assertIn("Allowed file types", unsupported.get_data(as_text=True))

        empty = self.client.post(
            "/process",
            data={"document": (io.BytesIO(b""), "empty.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(empty.status_code, 400)
        self.assertIn("file is empty", empty.get_data(as_text=True))

        oversized = self.client.post(
            "/process",
            data={
                "document": (
                    io.BytesIO(b"x" * (MAX_FILE_BYTES + 1)),
                    "large.txt",
                )
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(oversized.status_code, 413)
        self.assertIn("16 MB", oversized.get_data(as_text=True))

    def test_retry_reuses_uploaded_source_after_processing_failure(self):
        attempts = 0

        def fail_once(_path):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ValueError("Temporary text extraction failure.")
            return "Recovered text"

        with patch("Web.app.extract_text_document", side_effect=fail_once):
            response = self.client.post(
                "/process",
                data={
                    "document": (io.BytesIO(b"source"), "source.txt"),
                    "processing_mode": "as_is",
                },
                content_type="multipart/form-data",
            )
            job_id = self.job_id_from_response(response)
            self.wait_for_status(job_id, "failed")
            retry = self.client.post(f"/retry/{job_id}")
            self.assertEqual(retry.status_code, 200)
            completed = self.wait_for_status(job_id, "complete")

        self.assertEqual(attempts, 2)
        self.assertIn("Recovered text", self.client.get(completed["result_url"]).get_data(as_text=True))

    def test_vlm_request_locks_summary_language_to_english(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"status":"ok","result":"Summary"}'
        with patch("Web.app.urllib.request.urlopen", return_value=response) as urlopen:
            result = call_vlm_service("Source text")

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["language"], "en")
        self.assertEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
