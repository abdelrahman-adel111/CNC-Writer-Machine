import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLM.model_client import VLMClient

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
APP_BUILD_ID = "vlm-cpu-inference-v4"


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "vlm",
        "version": APP_BUILD_ID,
    })


@app.route("/analyze", methods=["POST"])  # JSON request: {"text": "...", "task": "summary", "language": "en"}
def analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    task = (data.get("task") or "summary").strip() or "summary"
    language = (data.get("language") or "en").strip() or "en"
    confidence_value = data.get("ocr_confidence", 1.0)
    if isinstance(confidence_value, bool) or not isinstance(confidence_value, (int, float)):
        confidence_value = 0.0
    ocr_confidence = max(0.0, min(float(confidence_value), 1.0))

    if not text:
        return jsonify({"status": "error", "message": "No text was provided to the VLM service."}), 400

    client = VLMClient(
        base_url=os.getenv("VLM_API_URL", ""),
        api_key=os.getenv("VLM_API_KEY", ""),
    )
    result = client.call_provider(
        text=text,
        task=task,
        language=language,
        ocr_confidence=ocr_confidence,
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=False, use_reloader=False, threaded=True)
