import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from VLM.app import APP_BUILD_ID, app
from VLM.model_client import (
    MAX_SUMMARY_INPUT_CHARS,
    OLLAMA_REQUEST_TIMEOUT_SECONDS,
    VLMClient,
)


class VLMModelClientTests(unittest.TestCase):
    def test_summary_request_forces_cpu_and_bounded_context(self):
        response = MagicMock()
        response.json.return_value = {"response": "A concise summary."}
        client = VLMClient(base_url="http://localhost:11434")

        with patch("requests.post", return_value=response) as post:
            result = client.call_provider("Document text", task="summary")

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["input_truncated"])
        self.assertEqual(post.call_args.kwargs["timeout"], OLLAMA_REQUEST_TIMEOUT_SECONDS)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["num_predict"], 80)
        self.assertEqual(payload["options"]["num_gpu"], 0)
        self.assertEqual(payload["options"]["num_ctx"], 2048)
        self.assertEqual(payload["options"]["num_batch"], 64)
        self.assertEqual(payload["keep_alive"], "1m")

    def test_health_reports_vlm_build_for_launcher_staleness_check(self):
        response = app.test_client().get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["version"], APP_BUILD_ID)
        self.assertEqual(APP_BUILD_ID, "vlm-cpu-inference-v4")

    def test_long_summary_input_keeps_excerpts_from_all_pages(self):
        text = "\n\n".join(
            f"[Page {page}]\nPage {page} opening content. " + ("details " * 35)
            for page in range(1, 41)
        )

        limited = VLMClient._limit_summary_input(text)

        self.assertLess(len(limited), MAX_SUMMARY_INPUT_CHARS + 100)
        self.assertIn("[Page 1]", limited)
        self.assertIn("[Page 40]", limited)
        self.assertIn("Representative excerpts", limited)

    def test_long_ollama_summary_reports_when_source_was_truncated(self):
        response = MagicMock()
        response.json.return_value = {"response": "A concise summary."}
        client = VLMClient(base_url="http://localhost:11434")
        long_text = "\n\n".join(
            f"[Page {page}]\n" + ("document content " * 50)
            for page in range(1, 21)
        )

        with patch("requests.post", return_value=response):
            result = client.call_provider(long_text, task="summary")

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["input_truncated"])

    def test_summary_input_budget_stays_bounded_for_very_long_documents(self):
        text = "\n\n".join(
            f"[Page {page}]\n" + ("document content " * 100)
            for page in range(1, 201)
        )

        limited = VLMClient._limit_summary_input(text)

        self.assertLessEqual(len(limited), MAX_SUMMARY_INPUT_CHARS + 100)
        self.assertIn("[Page 1]", limited)
        self.assertIn("[Page 200]", limited)


if __name__ == "__main__":
    unittest.main()
