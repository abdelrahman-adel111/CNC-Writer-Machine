# Real-image OCR evaluation sets

These manifests use fixed test examples from two external OCR datasets. Their
images and reference labels are separate from project uploads and the
synthetic benchmark.

## EvArEST real scene text

- Dataset: [EvArEST Arabic scene text recognition](https://huggingface.co/datasets/Melaraby/EvArEST-dataset-for-Arabic-scene-text-recognition)
- Dataset-listed license: BSD 3-Clause.
- Pinned dataset revision: `5bd10e4c5edd3f28bb81b1c11c9ccd48ee7b9322`.
- The manifest contains 20 Arabic and 20 English examples sampled with a fixed
  seed from the dataset's real Arabic and English test splits. Ground truth
  comes from each test row's `full_text`; the image bytes are stored locally
  for repeatable runs.
- These are cropped scene words, not complete photographed documents. They
  test recognition on real-world word crops but do not evaluate page-level
  text detection, forms, document layout, or full-image upload behavior.

## Historical Arabic manuscripts

- Dataset: [Historical Arabic Handwritten OCR](https://huggingface.co/datasets/sherif1313/Historical-Arabic-Handwritten-OCR)
- Dataset-listed license: Apache 2.0.
- Pinned dataset revision: `6de207a61155743c16c08937baaf2662a33153e5`.
- Eight full page images were selected at one-page intervals across the
  dataset's eight five-page book groups. Each image is paired with the
  dataset's expert-transcribed text file.
- This is a difficult historical handwriting/calligraphy domain. It is
  reported separately and is not representative of modern printed forms.

## Current comparison results

Run these commands from the project root:

```powershell
py -3.12 Script1\evaluate_ocr.py Script1\benchmark_evarest_samples.json --ocr paddleocr --threshold 0.90
py -3.12 Script1\evaluate_ocr.py Script1\benchmark_evarest_samples.json --ocr easyocr --threshold 0.90
py -3.12 Script1\evaluate_ocr.py Script1\benchmark_historical_manuscripts.json --ocr paddleocr --threshold 0.90
```

On the 40 identical EvArEST word crops, the initial PaddleOCR run measured
81.1% corpus character accuracy (84.6% Arabic, 78.5% English) and the
corrected EasyOCR run measured 80.2% (85.7% Arabic, 76.0% English). Both miss
the 90% target, and neither is a clear overall winner. The first EasyOCR run appeared to score 0% because
its adapter requested paragraph output, which omits confidence values, then
discarded those detections. The adapter now reads individual detections and
sorts them in reading order before collecting confidence.

The 20 English crops were also evaluated with the app's current mixed-language
configuration (`languages=["en", "ar"]`). PaddleOCR scored 81.8% character
accuracy and EasyOCR scored 76.0%. This is a focused mixed-mode check on
English-only crops, not a fully code-switched Arabic-English document test.

On the eight historical manuscript pages, PaddleOCR measured 39.7% corpus
character accuracy. The full EasyOCR page benchmark exceeded ten minutes
without completing and was stopped; no EasyOCR manuscript score is claimed.
The benchmark evaluator fails explicitly if a requested backend is not
initialized. Tesseract cannot currently be compared because its native
executable is not installed.

Model confidence is not a dependable proxy for character accuracy: PaddleOCR
gave sample `evarest-ar-0917` a 90.1% confidence score while its measured
character accuracy was only 80.0%.

These targeted samples are diagnostic comparisons, not evidence that the
project meets the 90% requirement on camera photographs or full documents.
The real-scene results cover cropped words only, while the manuscript set
covers historical handwriting only.
