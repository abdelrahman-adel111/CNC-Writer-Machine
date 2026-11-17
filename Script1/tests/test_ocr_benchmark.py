import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_ocr import error_rate, normalize
from script1 import DocumentProcessor


class OCRBenchmarkTests(unittest.TestCase):
    def test_paddle_cpu_uses_two_threads_without_onednn(self):
        with patch("script1.PaddleOCR", return_value=object()) as paddle:
            processor = DocumentProcessor(ocr_backend="paddleocr", languages=["en"])
            self.assertFalse(paddle.called)
            processor._initialize_paddle()

        options = paddle.call_args.kwargs
        self.assertEqual(options["cpu_threads"], 2)
        self.assertFalse(options["enable_mkldnn"])
        self.assertEqual(options["text_detection_model_name"], "PP-OCRv5_mobile_det")
        self.assertEqual(
            options["text_recognition_model_name"], "en_PP-OCRv5_mobile_rec"
        )
        self.assertNotIn("lang", options)

    def test_paddle_uses_arabic_mobile_recognizer_for_arabic(self):
        with patch("script1.PaddleOCR", return_value=object()) as paddle:
            processor = DocumentProcessor(ocr_backend="paddleocr", languages=["ar"])
            processor._initialize_paddle()

        options = paddle.call_args.kwargs
        self.assertEqual(options["text_detection_model_name"], "PP-OCRv5_mobile_det")
        self.assertEqual(
            options["text_recognition_model_name"], "arabic_PP-OCRv5_mobile_rec"
        )

    def test_auto_backend_falls_back_to_easyocr_when_paddle_fails(self):
        processor = DocumentProcessor(ocr_backend="auto", languages=["ar"])
        with patch("script1.PaddleOCR", side_effect=RuntimeError("Paddle unavailable")), patch.object(
            processor,
            "_initialize_easyocr",
            side_effect=lambda: setattr(processor, "easyocr_reader", object()),
        ):
            self.assertEqual(processor._get_ocr_backend(), "easyocr")

    def test_normalize_preserves_arabic_letters_and_english_case(self):
        self.assertEqual(normalize("  HELLO   Test "), "hello test")
        self.assertEqual(normalize("مُرَاجَعَة"), "مراجعة")

    def test_character_and_word_error_rates(self):
        self.assertEqual(error_rate("Hello OCR Test", "hello  OCR test"), 0.0)
        self.assertAlmostEqual(error_rate("cat", "cut"), 1 / 3)
        self.assertAlmostEqual(error_rate("a b", "a c", by_word=True), 0.5)

    def test_empty_reference_does_not_hide_insertions(self):
        self.assertEqual(error_rate("", "unexpected text"), 1.0)
        self.assertEqual(error_rate("", ""), 0.0)

    def test_native_pdf_text_is_extracted_without_ocr(self):
        document = pymupdf.open()
        page = document.new_page(width=612, height=792)
        page.insert_text((72, 72), "A digitally searchable English PDF lecture.")

        with tempfile.TemporaryDirectory(prefix="ocr_pdf_test_") as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            document.save(pdf_path)
            document.close()

            processor = DocumentProcessor(
                ocr_backend="paddleocr", languages=["auto"]
            )
            with patch.object(processor, "extract_text") as extract_text:
                result = processor.process_file(str(pdf_path))

        self.assertEqual(result["page_count"], 1)
        self.assertEqual(
            result["text"], "[Page 1]\nA digitally searchable English PDF lecture."
        )
        self.assertEqual(result["pages"][0]["backend"], "pdf_text")
        self.assertEqual(result["pages"][0]["confidence_source"], "pdf_text")
        self.assertEqual(result["detected_language"], "en")
        self.assertFalse(extract_text.called)
        self.assertFalse(processor._paddle_initialized)

    def test_scanned_pdf_page_is_rendered_and_sent_to_ocr(self):
        document = pymupdf.open()
        document.new_page(width=72, height=72)

        with tempfile.TemporaryDirectory(prefix="ocr_pdf_scan_test_") as temp_dir:
            pdf_path = Path(temp_dir) / "scan.pdf"
            document.save(pdf_path)
            document.close()

            processor = DocumentProcessor(
                ocr_backend="paddleocr", languages=["en"]
            )
            with patch.object(
                processor,
                "extract_text",
                return_value={
                    "text": "Scanned PDF page",
                    "confidence": 0.95,
                    "backend": "paddleocr",
                },
            ) as extract_text:
                result = processor.process_file(str(pdf_path))

        self.assertEqual(result["page_count"], 1)
        self.assertEqual(result["pages"][0]["backend"], "paddleocr")
        self.assertEqual(extract_text.call_args.args[0].shape, (300, 300, 3))

    def test_auto_language_detection_switches_from_arabic_to_english(self):
        processor = DocumentProcessor(
            ocr_backend="paddleocr", languages=["auto"]
        )
        processor._active_paddle_language = "ar"
        english_result = {
            "text": "English textbook page",
            "confidence": 0.95,
            "backend": "paddleocr",
        }
        with patch.object(
            processor, "extract_text", side_effect=[english_result, english_result]
        ) as extract_text, patch.object(
            processor,
            "_switch_paddle_language",
            side_effect=lambda language: setattr(
                processor, "_active_paddle_language", language
            ),
        ) as switch_language:
            result = processor.extract_text_auto(np.zeros((10, 10, 3), dtype=np.uint8))

        self.assertEqual(result["detected_language"], "en")
        switch_language.assert_called_once_with("en")
        self.assertEqual(extract_text.call_count, 2)

    def test_pdf_document_language_skips_redundant_auto_detection_pass(self):
        document = pymupdf.open()
        english_page = document.new_page(width=612, height=792)
        english_page.insert_text((72, 72), "A digitally searchable English lecture page.")
        scanned_page = document.new_page(width=72, height=72)
        scanned_page.draw_rect(pymupdf.Rect(0, 0, 72, 72), color=(0, 0, 0))

        with tempfile.TemporaryDirectory(prefix="ocr_pdf_auto_language_") as temp_dir:
            pdf_path = Path(temp_dir) / "mixed_source.pdf"
            document.save(pdf_path)
            document.close()

            processor = DocumentProcessor(
                ocr_backend="paddleocr", languages=["auto"]
            )
            processor._paddle_initialized = True
            processor._active_paddle_language = "ar"
            processor.paddleocr_reader = object()
            ocr_result = {
                "text": "Scanned English page",
                "confidence": 0.95,
                "backend": "paddleocr",
            }
            with patch.object(processor, "extract_text", return_value=ocr_result) as extract_text, patch.object(
                processor, "_switch_paddle_language"
            ) as switch_language:
                result = processor.process_file(str(pdf_path))

        self.assertEqual(result["detected_language"], "en")
        self.assertEqual(extract_text.call_count, 1)
        switch_language.assert_called_once_with("en")

    def test_detect_text_language_identifies_arabic_english_and_mixed(self):
        self.assertEqual(DocumentProcessor.detect_text_language("English text"), "en")
        self.assertEqual(DocumentProcessor.detect_text_language("نص عربي"), "ar")
        self.assertEqual(
            DocumentProcessor.detect_text_language("English نص عربي"), "mixed"
        )

    def test_arabic_boxes_are_sorted_reading_order(self):
        processor = object.__new__(DocumentProcessor)
        processor.languages = ["ar", "en"]
        boxes = [
            [[45, 0], [55, 0], [55, 12], [45, 12]],
            [[90, 0], [100, 0], [100, 12], [90, 12]],
            [[40, 35], [55, 35], [55, 47], [40, 47]],
        ]
        ordered = processor._order_ocr_boxes(
            ["left", "right", "next row"],
            [0.8, 0.9, 0.95],
            boxes,
        )
        self.assertEqual([text for text, _ in ordered], ["right", "left", "next row"])

    def test_easyocr_returns_text_confidence_and_rtl_order(self):
        processor = object.__new__(DocumentProcessor)
        processor.languages = ["ar"]
        processor.easyocr_reader = SimpleNamespace(
            readtext=lambda image, detail, paragraph: [
                ([[45, 0], [55, 0], [55, 12], [45, 12]], "left", 0.8),
                ([[90, 0], [100, 0], [100, 12], [90, 12]], "right", 0.9),
            ]
        )
        text, confidence = processor.ocr_easyocr(None)
        self.assertEqual(text, "right\nleft")
        self.assertAlmostEqual(confidence, 0.85)


if __name__ == "__main__":
    unittest.main()
