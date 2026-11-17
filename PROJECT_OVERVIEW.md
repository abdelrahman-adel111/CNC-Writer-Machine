# CNC Writing Machine OCR + VLM Demo

## Project goal
This project is a demonstration pipeline for a CNC writing machine concept. The system is designed to:

1. Accept an image or PDF document.
2. Extract text using OCR.
3. Send the extracted text to a local VLM/LLM service.
4. Show the result in a web interface.
5. Prepare the foundation for a future motion-control stage that can translate text into CNC writing commands.

---

## Architecture

### 1) OCR layer
File: `Script1/script1.py`

Role:
- Reads image/PDF input
- Preprocesses the document
- Tries OCR backends
- Returns extracted text

This is the layer responsible for converting a file into readable text.

### 2) Local AI layer
Files:
- `VLM/app.py`
- `VLM/model_client.py`

Role:
- Exposes a local API service
- Sends extracted text to Ollama-managed model
- Returns a summary/answer/translation depending on the task

This is the intelligence layer that interprets the OCR result.

### 3) Web interface
Files:
- `Web/app.py`
- `Web/templates/index.html`
- `Web/templates/result.html`
- `Web/static/style.css`

Role:
- Shows upload UI
- Accepts a file from the user
- Calls OCR processing
- Calls VLM analysis
- Displays OCR result + AI result in the browser

### 4) Startup helper
File: `start_demo.py`

Role:
- Starts VLM API and Flask web app together
- Makes it easier to run both services in one command

---

## Main files and purpose

### Script1/script1.py
Main OCR pipeline.

Responsibilities:
- Load PDF/image files
- Detect and rotate the page if needed
- Apply contrast and threshold preprocessing
- Try OCR engines in this order in `auto` mode: PaddleOCR, EasyOCR, then Tesseract
- Return extracted text and metadata

Important parts:
- `DocumentProcessor.__init__()`
- `preprocess_page()`
- `ocr_pytesseract()`
- `ocr_easyocr()`
- `ocr_paddleocr()`
- `extract_text()`
- `process_file()`

### OCR engines and current setup
The OCR script has integrations for **three OCR engines**. These are separate text-recognition engines; OpenCV preprocessing is not an OCR engine.

In `auto` mode:
1. **PaddleOCR** is preferred. With both `ar` and `en` requested, its Arabic recognition model is selected. It is the working engine tested with the current Python 3.12 setup and PaddleOCR 3.x.
2. **EasyOCR** is initialized only if PaddleOCR cannot initialize or as a fallback after PaddleOCR returns no valid text.
3. **Tesseract** is a final fallback through the `pytesseract` Python package, but it also requires the separate native Tesseract executable.

Important environment details from the tested setup:
- PaddleOCR successfully extracted `Hello OCR Test` from `demo_test.png`.
- The web app defaults to automatic language handling; Arabic/English overrides are available. VLM replies are set to Arabic when the extracted text contains Arabic characters.
- OCR first attempts the original image. For Arabic, Paddle text boxes are ordered top-to-bottom and right-to-left within each row so table/form labels do not get reversed. For a valid but less-confident read it also evaluates a deskewed/contrast-normalized version, then keeps the candidate with the higher model confidence. Only very low-confidence reads trigger additional contrast/adaptive-threshold variants; a high-confidence original avoids those extra passes.
- A test of `no_border_rotate_image.jpg`'s attached source image produced Arabic text with PaddleOCR at about 0.88 average confidence. This is a useful recognition result, but the photographed form still contains some OCR errors and should be proofread.
- The web app starts a fresh OCR child process for each upload so a native Paddle runtime crash cannot take down Flask or leave a persistent worker stuck on a later request. It sets the OCR worker's console encoding to UTF-8; without this, printing an Arabic OCR preview on a Windows legacy code page can fail even after recognition succeeded.
- The result page shows the original uploaded image or PDF beside the OCR output. Uploads are stored under random, non-user-controlled names for preview; previews older than 24 hours are cleaned up when a new upload arrives. Treat the displayed OCR confidence as a model score, not a guarantee of text accuracy.
- Searchable PDF pages are read from their embedded text directly and do not load an OCR model or render images. Scanned pages are rendered one at a time at 300 DPI with PyMuPDF; Poppler is not required, and all page images are no longer held in memory together. The former `pdf2image` path depended on the separate Poppler executable, which caused uploads to fail when it was missing from PATH.
- The AI summary prompt requires the selected output language, limits claims to the OCR text, and tells the model not to guess or silently repair ambiguous text. The OCR confidence is also passed to the AI service; if any page is below the 90% model-confidence gate, the app skips summarization and asks the user to review the text.
- OCR results are returned as soon as OCR is complete; the VLM summary continues in a separate background worker and updates the result page when ready. Ollama summary input is capped at 4,000 characters using page-spread excerpts for long documents, output is capped at 80 tokens, and context at 2,048 tokens. When excerpts are used, the result page says so. Inference is forced to CPU (`num_gpu=0`) because the local 4 GB MX130 saturated and stalled even an 8-token prompt; keeping the model loaded for one minute balances repeated requests against RAM use. A slow or unavailable model no longer blocks access to OCR results, and a request is abandoned after 90 seconds. The launcher checks both service build IDs so outdated running services are not reused silently.
- The upload form defaults to automatic language detection. Searchable PDF text is classified directly as Arabic, English, or mixed. When a scanned page belongs to a PDF whose embedded text already establishes a single language, that language selects the OCR model directly and avoids a redundant low-resolution detection pass. Image-only PDFs without a detectable language still use a small OCR probe before full-resolution OCR. Mixed scans use the Arabic recognizer, while direct-text PDFs preserve both scripts; explicit Arabic/English overrides remain available. OCR confidence remains a model score, not verified character accuracy.
- EasyOCR's paragraph mode omits per-line confidence values; the adapter now requests individual detections and restores reading order before calculating confidence.
- OCR confidence is not ground-truth accuracy. The web app uses a conservative gate: every page must have a model confidence of at least 90% before the AI summary is requested. This is a review gate only, not a guarantee that the document reached 90% character accuracy. All results require an explicit human review/approval; the current approval is held in the running Flask process and does not send anything to hardware.
- `Script1/evaluate_ocr.py` measures character error rate (CER) and word error rate (WER) against reference text and reports corpus plus per-language/category accuracy. It errors explicitly if a requested backend could not initialize. The small project-owned set is `Script1/benchmark_samples.json`; run it with `py -3.12 Script1/evaluate_ocr.py --ocr paddleocr --threshold 0.90` from the project root. External tests are separate: ten synthetic printed pages in `Script1/benchmark_external_samples.json`; forty real Arabic/English scene-word crops in `Script1/benchmark_evarest_samples.json`; twenty English scene-word crops run using the current mixed-language setting in `Script1/benchmark_evarest_mixed_english_samples.json`; and eight complete pages from an expert-transcribed historical Arabic manuscript set in `Script1/benchmark_historical_manuscripts.json`. Run each manifest with the same command, changing `--ocr paddleocr` to `--ocr easyocr` to compare engines on identical samples. The scene-word sets evaluate cropped-word recognition, not page-level detection; the mixed-language check contains English-only crops, not code-switched documents; and the manuscript test is a separate handwritten challenge domain. These targeted tests do not substitute for representative current camera photos, forms, and PDFs. The previous synthetic benchmark scored 61.4% character accuracy and remains a separate baseline. See `Script1/benchmark_real_sources.md` and `Script1/benchmark_external_sources.md` for sources and limitations. Model confidence must not be reported as ground-truth accuracy.
- The upload now starts a background processing job and the browser polls its real current stage (file received, OCR, AI model, review) rather than showing a timer-based progress animation. Job state and review approval are in memory, so they are intended for the current single-process local demo.
- PaddleOCR initializes lazily, so a searchable PDF does not pay model startup time. It runs on CPU with two inference threads and oneDNN disabled in this setup. English-only OCR uses PP-OCRv5 mobile detection and English mobile recognition instead of PaddleOCR's default PP-OCRv6 medium models. On the supplied 502x652 textbook page, the previous PP-OCRv6 medium setup took 116.3 seconds for OCR after initialization; the explicit mobile pair took 42.8 seconds after a 2.4-second cached initialization and extracted the page's main text at 0.984 confidence. This one-page timing and model confidence do not prove ground-truth accuracy, but the inspected output retained the main paragraphs. On `demo_test.png`, two threads reduced OCR time from 19.8 to 16.7 seconds (about 15%) with the same extracted text and effectively identical confidence; four threads were slower, and enabling oneDNN returned no text. These are measurements on individual English samples, not guarantees for every document. The installed Paddle build is CPU-only, so the NVIDIA MX130 is not currently used. A Windows worker exit code of `3221225477` (`0xC0000005`) was observed during Arabic OCR; that is a native access violation, not evidence by itself of out-of-memory. At the time of diagnosis the machine reported 15.8 GB total and 10.3 GB free memory. The web worker retries that specific native crash once, and isolation keeps Flask running if the OCR process still fails.
- Arabic OCR uses PP-OCRv5 mobile detector and Arabic mobile recognizer (instead of the slower server detector). The worker no longer spends an additional inference pass warming the model on a blank image. Earlier measurements with a reused process were about 2 seconds for initialization and 14-15 seconds for inference. Reusing the worker between web requests proved unreliable: the second upload could hang, so the web app now launches a fresh process for every upload. The current 0.92 early-accept threshold processed the uploaded Arabic form in about 28 seconds through the web worker, selecting the original-image result at 0.93 confidence (703 OCR characters, plus the page label). A direct processor run including RTL ordering took about 15 seconds. Lower-confidence images run additional preprocessing candidates, so their processing may take longer.
- Scanned PDF worker timeouts scale with the number of pages that require OCR (up to 30 minutes); the count uses the same text-coverage test as extraction, so searchable pages are not mistakenly counted. A worker that stalls is still isolated from Flask.
- The Tesseract executable was not found on PATH or in the checked standard installation locations, so that fallback is not currently usable.
- The EasyOCR fallback may need its model files to finish downloading and initialize successfully.
- PaddleOCR uses one recognition language model per processor. When both English and Arabic are requested, Arabic is prioritized; mixed-language or English-only files may need a language selector or a separate English pass for best results.
- PaddleOCR 3.x uses a different API from older PaddleOCR releases. The script uses the 3.x prediction result format and disables oneDNN acceleration because the installed Paddle runtime produced an inference error with that path.

OCR fallback does not mean all engines run every time. The script starts with the preferred engine, tries alternate image preprocessing variants, and moves to fallback engines only when needed. This avoids unnecessarily loading and running every OCR model for each upload.

### VLM/app.py
Local Flask API for the VLM layer.

Responsibilities:
- Receive JSON with text + task
- Call `VLMClient.call_provider()`
- Return response to the web app

### VLM/model_client.py
Handles local model connection.

Responsibilities:
- Connect to Ollama on `http://localhost:11434`
- Use model `qwen2.5vl:3b`
- Send prompt to the model
- Return result as JSON

### Web/app.py
Main web backend.

Responsibilities:
- Accept uploads
- Save temp file
- Invoke OCR processing
- Invoke VLM analysis
- Render result page

### Web/templates/index.html
Upload page.

### Web/templates/result.html
Displays:
- file name
- page count
- OCR backend
- summary
- AI result
- full extracted text

### Web/static/style.css
Theme and layout styling for the demo.

---

## Workflow

### Normal flow
1. User opens the web app.
2. User uploads an image or PDF.
3. Flask app saves the file as a temporary upload.
4. `DocumentProcessor` reads the file.
5. OCR tries to extract readable text.
6. If text is found, it is sent to the local VLM.
7. The VLM returns a summary/answer.
8. The result page shows OCR result and AI result.

### Example data flow
`image/PDF -> OCR script -> extracted text -> VLM -> summary/answer -> web page`

---

## Important fact
The VLM is not designed to read the file directly in the current architecture.

The pipeline is:
- OCR first
- then AI reasoning on the extracted text

This is the correct structure for a document-processing and analysis demo.

---

## Current limitations

### OCR limitations
- OCR often fails on:
  - low-contrast images
  - blurred images
  - noisy screenshots
  - non-document images
  - poor scans
- First run may be slow because the model is downloading or initializing.
- Tesseract must be installed on the system for the PyTesseract route to work reliably.

### VLM limitations
- The generated model is local and lightweight.
- Performance is acceptable for demos, not heavy production workloads.
- It works best when the OCR text is clean and meaningful.

### Hardware layer
This project currently does not include the CNC motion-controller stage yet.
That is a future script/module for controlling motors and writing output physically.

---

## Typical issues and causes

### OCR backend: none
This means the OCR engines did not return valid text, or no engine could initialize. The result page now displays an OCR diagnostic and a message in Full Text instead of leaving that field blank.

Common causes:
- image is too faint or dark
- poor contrast
- image is noisy or blurry
- file is not a real document scan
- OCR dependencies/models are not ready
- an OCR engine API/runtime mismatch
- the Tesseract Python package is installed but its native executable is missing

### VLM timed out
This usually happens when:
- the OCR result is empty
- the model is still initializing
- the request took too long on the machine

### Flask page does not load
This usually means:
- the web app is not running
- VLM service is not running
- debug reloader restarted the process unexpectedly

---

## Recommended environment
Use a stable Python environment such as Python 3.12 for OCR-heavy projects.

Why:
- some OCR libraries behave more reliably there
- avoids version problems seen with newer Python installs

---

## Recommended next steps
1. Install Tesseract on Windows.
2. Confirm the OCR backend path is valid.
3. Test with a clear black-text-on-white-image.
4. Confirm the VLM service is running before using the web app.
5. Improve preprocessing and image quality before OCR.
6. Add a future motion-control script for the CNC hardware.

---

## Summary
This project is a working OCR + VLM + web demo structure.

The core idea is:
- read file
- extract text
- analyze text locally
- show result in browser

The current bottleneck is OCR quality and environment readiness, not the web architecture itself.
