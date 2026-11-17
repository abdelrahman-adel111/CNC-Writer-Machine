# External document OCR benchmark

`benchmark_document_samples.json` and `benchmark_document_images/` contain ten
selected pages from **Misraj-DocOCR**:

- Dataset: https://huggingface.co/datasets/Misraj/Misraj-DocOCR
- Dataset card describes 400 expert-verified pages, a mix of real and
  synthetic documents, including books, reports, forms, and complex layouts.
- The Hugging Face dataset metadata declares Apache-2.0. The source row and
  UUID are recorded per sample.

The selected pages include Arabic book scans, a bordered page, a bilingual
page, report-like pages, a page with a photograph, and tables. The dataset does
not identify real versus synthetic provenance per row, so no individual page
is represented here as definitively real. It is closer to published document
scans than the synthetic-only set, but does not measure phone-camera capture
conditions such as blur, perspective, glare, or uneven lighting.

`benchmark_ocr_engines.py` compares engines on the exact same original RGB
images using a single OCR pass, without preprocessing variants. This isolates
recognizer behavior but is not a full production-pipeline comparison. Run:

```powershell
py -3.12 Script1\benchmark_ocr_engines.py --engines paddleocr easyocr --source-rows 0 60 100 180 380
```

Tesseract is included in the comparison tool but is reported as unavailable
unless its native executable is installed. PaddleOCR and EasyOCR scores are
small-sample measurements, not a 90% acceptance claim. For final deployment
validation, add representative, permission-cleared camera photographs and
manually verified references from the intended hardware environment.
