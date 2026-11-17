# External OCR benchmark samples

The images and reference text in `benchmark_external_images/` and
`benchmark_external_samples.json` are selected from **Arabic OCR Synthetic
Scans (Faker 300k)** by Loay Amin:

- Dataset: https://huggingface.co/datasets/loay/arabic-ocr-synthetic-scans-faker-300k
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Dataset card describes the images as synthetic scanned pages and the content
  as generated Arabic text. The content has no semantic meaning.

The ten selected rows cover clean, light, moderate, and heavy degradation,
single-column text, poetry, a title page, and an index. One row is tagged by
the dataset as containing sparse English. Reference text comes from the
dataset's ground-truth `text` field; structural XML-style tags were removed and
whitespace was normalized for the text-only CER/WER comparison. The source
dataset row and file name are retained in each manifest entry.

This is a reproducible initial OCR check, not a representative real-world
accuracy claim. The generated pages do not substitute for camera photographs,
real forms, mixed-script technical diagrams, or PDFs. Do not infer the
project's 90% accuracy target from this synthetic subset.

With the project's current PaddleOCR pipeline, the ten-sample run measured
61.4% corpus character accuracy (38.6% CER); one sample met the 90% target.
The benchmark intentionally reports this shortfall rather than treating model
confidence as accuracy.
