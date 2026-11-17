import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

try:
    from .pdf_utils import has_extractable_pdf_text
except ImportError:
    from pdf_utils import has_extractable_pdf_text

try:
    import pymupdf
except ImportError:
    pymupdf = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import easyocr
except Exception:
    easyocr = None

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class DocumentProcessor:
    MIN_TEXT_CHARS = 3
    MIN_CONFIDENCE = 0.35

    def __init__(self, ocr_backend: str = "auto", languages: Optional[List[str]] = None):
        self.ocr_backend = self._resolve_backend(ocr_backend)
        self.languages = languages or ["en", "ar"]
        self.easyocr_reader = None
        self.paddleocr_reader = None
        self._paddle_initialized = False
        self._active_paddle_language = None
        self._auto_detect_language = (
            "auto" in self.languages
            or {"ar", "en"}.issubset(set(self.languages))
        )
        self.tesseract_path = self._detect_tesseract()
        self.backend_errors: Dict[str, str] = {}

        if "auto" in self.languages:
            self.languages = ["ar"]

        if self.ocr_backend == "easyocr" or (
            self.ocr_backend == "auto" and PaddleOCR is None
        ):
            self._initialize_easyocr()

        if pytesseract is not None and not self.tesseract_path:
            self.backend_errors["tesseract"] = (
                "The Python pytesseract package is installed, but the Tesseract executable "
                "was not found. Install Tesseract OCR or set TESSERACT_CMD."
            )

    def _initialize_paddle(self, language: Optional[str] = None) -> None:
        if self._paddle_initialized:
            return
        self._paddle_initialized = True
        if self.ocr_backend not in {"paddleocr", "auto"} or PaddleOCR is None:
            return

        lang = language or (
            "ar" if "ar" in self.languages else (
                "en" if "en" in self.languages else "ch"
            )
        )
        self._active_paddle_language = lang
        try:
            paddle_options = {
                "device": "cpu",
                "cpu_threads": 2,
                "enable_cinn": False,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "enable_mkldnn": False,
            }
            if lang == "ar":
                paddle_options.update({
                    "text_detection_model_name": "PP-OCRv5_mobile_det",
                    "text_recognition_model_name": "arabic_PP-OCRv5_mobile_rec",
                })
            elif lang == "en":
                paddle_options.update({
                    "text_detection_model_name": "PP-OCRv5_mobile_det",
                    "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
                })
            else:
                paddle_options["lang"] = lang
            self.paddleocr_reader = PaddleOCR(**paddle_options)
        except TypeError:
            try:
                self.paddleocr_reader = PaddleOCR(
                    use_angle_cls=True, lang=lang, show_log=False
                )
            except Exception as exc:
                self.paddleocr_reader = None
                self.backend_errors["paddleocr"] = str(exc)
        except Exception as exc:
            self.paddleocr_reader = None
            self.backend_errors["paddleocr"] = str(exc)

    def _switch_paddle_language(self, language: str) -> None:
        if language == self._active_paddle_language:
            return
        self.languages = [language]
        self.paddleocr_reader = None
        self._paddle_initialized = False
        self._initialize_paddle(language)

    def _initialize_easyocr(self) -> None:
        if self.easyocr_reader is not None or easyocr is None:
            return
        try:
            self.easyocr_reader = easyocr.Reader(self.languages, gpu=False)
        except Exception as exc:
            self.easyocr_reader = None
            self.backend_errors["easyocr"] = str(exc)

    # ------------------------------------------------------------------
    # Backend detection / warmup
    # ------------------------------------------------------------------
    def _detect_tesseract(self) -> Optional[str]:
        candidates = [
            os.environ.get("TESSERACT_CMD"),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
            "tesseract",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            if candidate.strip() == "tesseract":
                try:
                    import shutil
                    found = shutil.which("tesseract")
                    if found:
                        return found
                except Exception:
                    pass
            elif os.path.exists(candidate):
                return candidate
        return None

    def warmup(self) -> None:
        """تهيئة الموديلات مرة واحدة على صورة صغيرة عشان أول استدعاء حقيقي ميقعش."""
        try:
            if self.paddleocr_reader is not None:
                self.ocr_paddleocr(np.zeros((160, 900, 3), dtype=np.uint8))
        except Exception as exc:
            self.backend_errors["paddleocr"] = str(exc)
        try:
            if self.easyocr_reader is not None:
                self.easyocr_reader.readtext(np.zeros((160, 900, 3), dtype=np.uint8), detail=0)
        except Exception as exc:
            self.backend_errors["easyocr"] = str(exc)
        try:
            if pytesseract is not None and self.tesseract_path:
                if self.tesseract_path != "tesseract":
                    pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
                img = Image.fromarray(np.full((120, 700, 3), 255, dtype=np.uint8))
                pytesseract.image_to_string(img, config="--psm 6")
        except Exception as exc:
            self.backend_errors["tesseract"] = str(exc)

    def _record_backend_error(self, name: str, exc: Exception) -> None:
        self.backend_errors[name] = str(exc)

    def _diagnostic(self, text_detected: bool) -> Optional[str]:
        if text_detected:
            return None
        details = "; ".join(
            f"{name}: {message}" for name, message in self.backend_errors.items()
        )
        if self._get_ocr_backend() == "none":
            return details or (
                "No OCR backend is available. Install Tesseract OCR or ensure EasyOCR/PaddleOCR "
                "models are installed and initialized."
            )
        if details:
            return f"OCR returned no text. Backend errors: {details}"
        return "OCR engines ran but returned no text for this image."

    def _resolve_backend(self, requested: str) -> str:
        requested = (requested or "auto").lower().strip()
        if requested in {"easyocr", "pytesseract", "tesseract", "paddleocr", "auto"}:
            return requested
        return "auto"

    def _get_ocr_backend(self) -> str:
        """يرجّع الـ backend المستقر أولاً، ثم الـ fallback."""
        preferred = ["paddleocr", "easyocr", "pytesseract"]

        if self.ocr_backend in {"paddleocr", "auto"} and not self._paddle_initialized:
            self._initialize_paddle()
        if (
            self.ocr_backend == "auto"
            and self.paddleocr_reader is None
            and self.easyocr_reader is None
        ):
            self._initialize_easyocr()

        if self.ocr_backend in {"paddleocr", "easyocr", "pytesseract", "tesseract"}:
            requested = "pytesseract" if self.ocr_backend == "tesseract" else self.ocr_backend
            if requested == "pytesseract" and pytesseract is not None and self.tesseract_path:
                return "pytesseract"
            if requested == "easyocr" and self.easyocr_reader is not None:
                return "easyocr"
            if requested == "paddleocr" and self.paddleocr_reader is not None:
                return "paddleocr"

        for name in preferred:
            if name == "pytesseract" and pytesseract is not None and self.tesseract_path:
                return "pytesseract"
            if name == "easyocr" and self.easyocr_reader is not None:
                return "easyocr"
            if name == "paddleocr" and self.paddleocr_reader is not None:
                return "paddleocr"
        return "none"

    def _call_backend(self, name: str, image: np.ndarray) -> Tuple[str, float]:
        if name == "paddleocr":
            return self.ocr_paddleocr(image)
        if name == "easyocr":
            return self.ocr_easyocr(image)
        if name == "pytesseract":
            return self.ocr_pytesseract(image)
        return "", 0.0

    # ------------------------------------------------------------------
    # File reading
    # ------------------------------------------------------------------
    def read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return img

    def read_pdf(self, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        if pymupdf is None:
            raise RuntimeError(
                "PDF rendering requires PyMuPDF. Install it with "
                "`py -3.12 -m pip install -r Web/requirements.txt`."
            )

        scale = dpi / 72
        matrix = pymupdf.Matrix(scale, scale)
        images = []
        with pymupdf.open(pdf_path) as document:
            for page in document:
                images.append(self._render_pdf_page(page, matrix))
        return images

    @staticmethod
    def _render_pdf_page(page, matrix: Optional[object] = None) -> np.ndarray:
        if matrix is None:
            matrix = pymupdf.Matrix(300 / 72, 300 / 72)
        pixmap = page.get_pixmap(
            matrix=matrix, colorspace=pymupdf.csRGB, alpha=False
        )
        rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, 3
        )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def detect_text_language(text: str) -> str:
        arabic_count = sum(
            "\u0600" <= char <= "\u06ff"
            or "\u0750" <= char <= "\u077f"
            or "\ufb50" <= char <= "\ufdff"
            or "\ufe70" <= char <= "\ufeff"
            for char in text
        )
        latin_count = sum(char.isascii() and char.isalpha() for char in text)
        if arabic_count and latin_count:
            return "mixed"
        if arabic_count:
            return "ar"
        if latin_count:
            return "en"
        return "unknown"

    @staticmethod
    def _has_extractable_pdf_text(page, text: str) -> bool:
        return has_extractable_pdf_text(page, text)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def rotate_image(self, image: np.ndarray, angle_deg: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(
            image, matrix, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )

    def auto_rotate_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if pytesseract is not None:
            try:
                osd = pytesseract.image_to_osd(Image.fromarray(gray))
                for line in osd.splitlines():
                    if "Rotate" in line:
                        try:
                            angle = int(line.split(":")[-1].strip())
                            if angle in {90, 180, 270}:
                                return self.rotate_image(image, angle)
                        except Exception:
                            pass
            except Exception:
                pass

        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 1000:
            return image

        rect = cv2.minAreaRect(largest)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        if abs(angle) > 1:
            return self.rotate_image(image, angle)
        return image

    def largest_page_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 5000:
            return None

        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return np.array(box, dtype=np.float32)

    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        max_width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        max_height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (max_width, max_height))

    def preprocess_page(self, image: np.ndarray) -> np.ndarray:
        """بترجع صورة RGB جاهزة للـ OCR."""
        image = self.auto_rotate_image(image)

        contour = self.largest_page_contour(image)
        if contour is not None:
            try:
                image = self.four_point_transform(image, contour)
            except Exception:
                pass

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        final = binary if cv2.countNonZero(binary) > cv2.countNonZero(binary_inv) else binary_inv
        final = cv2.medianBlur(final, 3)
        return cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)

    def _build_variants(self, image: np.ndarray):
        """Generate contrast-enhanced and adaptive-thresholded RGB variants."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        enhanced = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8)
        ).apply(gray)
        yield cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        adaptive = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        yield cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)

    # ------------------------------------------------------------------
    # OCR backends
    # ------------------------------------------------------------------
    def ocr_easyocr(self, image: np.ndarray) -> Tuple[str, float]:
        if self.easyocr_reader is None:
            return "", 0.0
        try:
            results = self.easyocr_reader.readtext(
                image, detail=1, paragraph=False
            )
            texts, confidences, boxes = [], [], []
            for item in results:
                if len(item) >= 3:
                    box, text, conf = item[0], item[1], item[2]
                    if text and text.strip():
                        boxes.append(box)
                        texts.append(text.strip())
                        confidences.append(float(conf))
            ordered = self._order_ocr_boxes(texts, confidences, boxes)
            text = "\n".join(item[0] for item in ordered).strip()
            confidence = float(np.mean([item[1] for item in ordered])) if ordered else 0.0
            return text, confidence
        except Exception as exc:
            self._record_backend_error("easyocr", exc)
            return "", 0.0

    def ocr_pytesseract(self, image: np.ndarray) -> Tuple[str, float]:
        if pytesseract is None or not self.tesseract_path:
            return "", 0.0
        try:
            if self.tesseract_path != "tesseract":
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            # image جاية RGB من preprocess_page -> PIL يقبلها مباشرة
            pil_img = Image.fromarray(image)
            config = "--psm 6"
            text = pytesseract.image_to_string(pil_img, config=config)
            cleaned = "\n".join(
                line.strip() for line in text.splitlines() if line.strip()
            ).strip()

            try:
                data = pytesseract.image_to_data(
                    pil_img, config=config, output_type=pytesseract.Output.DICT
                )
                confs = []
                for c, t in zip(data["conf"], data["text"]):
                    try:
                        ci = int(c)
                    except Exception:
                        continue
                    if t and t.strip() and ci >= 0:
                        confs.append(ci)
                conf = (sum(confs) / len(confs)) / 100.0 if confs else 0.0
            except Exception:
                conf = 0.0

            return cleaned, conf
        except Exception as exc:
            self._record_backend_error("tesseract", exc)
            return "", 0.0

    def ocr_paddleocr(self, image: np.ndarray) -> Tuple[str, float]:
        if self.paddleocr_reader is None:
            return "", 0.0
        try:
            if hasattr(self.paddleocr_reader, "predict"):
                results = self.paddleocr_reader.predict(image)
                text_parts, confidences = [], []
                for result in results or []:
                    if hasattr(result, "json"):
                        result = result.json
                    if not isinstance(result, dict):
                        continue
                    data = result.get("res", result)
                    if not isinstance(data, dict):
                        continue
                    texts = data.get("rec_texts", [])
                    scores = data.get("rec_scores", [])
                    boxes = data.get("rec_polys")
                    if boxes is None or len(boxes) != len(texts):
                        boxes = data.get("dt_polys")
                    ordered = self._order_ocr_boxes(texts, scores, boxes)
                    for text, confidence in ordered:
                        if text and str(text).strip():
                            text_parts.append(str(text).strip())
                            if confidence is not None:
                                confidences.append(confidence)
                if text_parts:
                    confidence = (
                        float(sum(confidences) / len(confidences))
                        if confidences else 0.0
                    )
                    return "\n".join(text_parts), confidence
                return "", 0.0

            result = self.paddleocr_reader.ocr(image, cls=True)
            text_parts, confidences = [], []
            if result and result[0]:
                lines = result[0]
                ordered = self._order_ocr_boxes(
                    [line[1][0] if line and len(line) >= 2 else "" for line in lines],
                    [line[1][1] if line and len(line) >= 2 else 0.0 for line in lines],
                    [line[0] if line and len(line) >= 2 else [] for line in lines],
                )
                for text, confidence in ordered:
                    if text and text.strip():
                        text_parts.append(text)
                        confidences.append(confidence or 0.0)
            if not text_parts:
                return "", 0.0
            return "\n".join(text_parts).strip(), float(sum(confidences) / len(confidences))
        except Exception as exc:
            self._record_backend_error("paddleocr", exc)
            return "", 0.0

    def _order_ocr_boxes(self, texts, scores, boxes):
        entries = []
        if boxes is None or len(boxes) != len(texts) or len(boxes) == 0:
            return [
                (str(text), float(scores[index]) if index < len(scores) else None)
                for index, text in enumerate(texts)
            ]

        for index, text in enumerate(texts):
            if not text or not str(text).strip():
                continue
            try:
                points = np.asarray(boxes[index], dtype=np.float32).reshape(-1, 2)
                if len(points) < 4:
                    raise ValueError("OCR box has fewer than four points")
                x_center = float(np.mean(points[:, 0]))
                y_center = float(np.mean(points[:, 1]))
                height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
            except (TypeError, ValueError):
                return [
                    (str(item), float(scores[i]) if i < len(scores) else None)
                    for i, item in enumerate(texts)
                ]

            confidence = float(scores[index]) if index < len(scores) else None
            entries.append({
                "text": str(text),
                "confidence": confidence,
                "x": x_center,
                "y": y_center,
                "height": height,
            })

        if len(entries) < 2:
            return [(entry["text"], entry["confidence"]) for entry in entries]

        row_tolerance = max(
            12.0,
            float(np.median([entry["height"] for entry in entries])) * 0.65,
        )
        rows = []
        for entry in sorted(entries, key=lambda item: item["y"]):
            row = min(
                rows,
                key=lambda candidate: abs(candidate["center"] - entry["y"]),
                default=None,
            )
            if row is None or abs(row["center"] - entry["y"]) > row_tolerance:
                rows.append({"center": entry["y"], "entries": [entry]})
                continue

            row["entries"].append(entry)
            row["center"] = sum(item["y"] for item in row["entries"]) / len(row["entries"])

        rtl = "ar" in self.languages
        ordered = []
        for row in sorted(rows, key=lambda item: item["center"]):
            ordered.extend(sorted(row["entries"], key=lambda item: item["x"], reverse=rtl))
        return [(entry["text"], entry["confidence"]) for entry in ordered]

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------
    def _is_valid(self, text: str, conf: float) -> bool:
        if not text or len(text.strip()) < self.MIN_TEXT_CHARS:
            return False
        if conf > 0 and conf < self.MIN_CONFIDENCE:
            return False
        if not any(ch.isalnum() for ch in text):
            return False
        return True

    def extract_text(self, image: np.ndarray) -> Dict[str, object]:
        backend = self._get_ocr_backend()
        if backend == "none":
            return {"text": "", "confidence": 0.0, "backend": "none"}

        original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        candidates = []

        def try_candidate(candidate: np.ndarray) -> float:
            text, confidence = self._call_backend(backend, candidate)
            if self._is_valid(text, confidence):
                candidates.append({
                    "text": text,
                    "confidence": confidence,
                    "backend": backend,
                })
            return confidence

        original_confidence = try_candidate(original)
        if candidates and original_confidence >= 0.92:
            return max(candidates, key=lambda item: item["confidence"])

        prepared = self.preprocess_page(image)
        try_candidate(prepared)

        best_confidence = max(
            (item["confidence"] for item in candidates), default=0.0
        )
        if best_confidence < 0.82:
            for variant in self._build_variants(image):
                try_candidate(variant)

        if candidates:
            return max(candidates, key=lambda item: item["confidence"])

        # 3) fallback: باقي الـ backends على prepared (auto فقط)
        if self.ocr_backend == "auto":
            for name in ("paddleocr", "easyocr", "pytesseract"):
                if name == backend:
                    continue
                if name == "easyocr" and self.easyocr_reader is None:
                    self._initialize_easyocr()
                for candidate in (original, prepared):
                    text, conf = self._call_backend(name, candidate)
                    if self._is_valid(text, conf):
                        return {"text": text, "confidence": conf, "backend": name}

        return {"text": "", "confidence": 0.0, "backend": "none"}

    def extract_text_auto(
        self, image: np.ndarray, preferred_language: Optional[str] = None
    ) -> Dict[str, object]:
        if preferred_language in {"ar", "en"} and self._auto_detect_language:
            if (
                self.ocr_backend in {"paddleocr", "auto"}
                and PaddleOCR is not None
                and not self._paddle_initialized
            ):
                self._initialize_paddle(preferred_language)
            backend = self._get_ocr_backend()
            if backend == "paddleocr":
                self._switch_paddle_language(preferred_language)
            elif backend == "easyocr" and preferred_language not in self.languages:
                self.languages = [preferred_language]
                self.easyocr_reader = None
                self._initialize_easyocr()
            result = self.extract_text(image)
            detected_language = self.detect_text_language(str(result["text"]))
            result["detected_language"] = (
                detected_language if detected_language != "unknown"
                else preferred_language
            )
            return result

        height, width = image.shape[:2]
        scale = min(0.5, 1000 / max(height, width))
        probe = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        result = self.extract_text(probe)
        detected_language = self.detect_text_language(str(result["text"]))
        selected_language = (
            "ar" if detected_language in {"ar", "mixed"} else (
                "en" if detected_language == "en" else None
            )
        )
        if (
            self._auto_detect_language
            and result["backend"] == "paddleocr"
        ):
            if selected_language:
                self._switch_paddle_language(selected_language)
            result = self.extract_text(image)
            detected_language = self.detect_text_language(str(result["text"]))
        result["detected_language"] = detected_language
        return result

    # ------------------------------------------------------------------
    # File processing
    # ------------------------------------------------------------------
    def process_file(self, source_path: str) -> Dict[str, object]:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")

        suffix = path.suffix.lower()
        pages: List[Dict[str, object]] = []
        combined_text: List[str] = []

        def append_result(idx: int, result: Dict[str, object], source: str) -> None:
            extracted_text = str(result["text"]).strip()
            backend = str(result["backend"])
            ok = backend != "none" and bool(extracted_text)
            if ok:
                combined_text.append(f"[Page {idx}]\n{extracted_text}")
            pages.append({
                "page": idx,
                "backend": backend if ok else "none",
                "confidence": float(result["confidence"]) if ok else 0.0,
                "confidence_source": (
                    "pdf_text" if backend == "pdf_text" else source
                ),
                "detected_language": result.get(
                    "detected_language", self.detect_text_language(extracted_text)
                ),
                "text": extracted_text if ok else "",
            })

        if suffix == ".pdf":
            if pymupdf is None:
                raise RuntimeError(
                    "PDF rendering requires PyMuPDF. Install it with "
                    "`py -3.12 -m pip install -r Web/requirements.txt`."
                )
            with pymupdf.open(str(path)) as document:
                pdf_pages = [
                    (page, page.get_text("text", sort=True).strip())
                    for page in document
                ]
                document_language = self.detect_text_language(
                    "\n".join(text for _, text in pdf_pages)
                )
                preferred_document_language = (
                    document_language
                    if document_language in {"ar", "en"}
                    else None
                )
                for idx, (pdf_page, pdf_text) in enumerate(pdf_pages, start=1):
                    if self._has_extractable_pdf_text(pdf_page, pdf_text):
                        append_result(idx, {
                            "text": pdf_text,
                            "confidence": 0.0,
                            "backend": "pdf_text",
                            "detected_language": self.detect_text_language(pdf_text),
                        }, "pdf_text")
                        continue

                    page_image = self._render_pdf_page(pdf_page)
                    page_language = self.detect_text_language(pdf_text)
                    preferred_language = (
                        page_language
                        if page_language in {"ar", "en"}
                        else preferred_document_language
                    )
                    result = (
                        self.extract_text_auto(page_image, preferred_language)
                        if self._auto_detect_language else self.extract_text(page_image)
                    )
                    append_result(idx, result, "ocr")
        elif suffix in SUPPORTED_IMAGE_EXTS:
            image = self.read_image(str(path))
            result = (
                self.extract_text_auto(image)
                if self._auto_detect_language else self.extract_text(image)
            )
            append_result(1, result, "ocr")
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use PDF or an image file.")

        final_text = "\n\n".join(combined_text).strip()
        any_ok = any(p["backend"] != "none" for p in pages)

        diagnostic = self._diagnostic(any_ok)
        detected_languages = {
            page["detected_language"] for page in pages
            if page["detected_language"] in {"ar", "en", "mixed"}
        }
        detected_language = (
            "mixed" if "mixed" in detected_languages or len(detected_languages) > 1
            else next(iter(detected_languages), "unknown")
        )
        return {
            "status": "ok",
            "file": str(path),
            "page_count": len(pages),
            "pages": pages,
            "text": final_text,
            "detected_language": detected_language,
            "ocr_status": "ready" if any_ok else "no_text_detected",
            "warning": diagnostic,
            "backend_errors": self.backend_errors.copy(),
        }


# ---------------------------------------------------------------------------
# FastAPI integration example:
#
# from fastapi import FastAPI, UploadFile, File
# from tempfile import NamedTemporaryFile
# app = FastAPI()
# processor = DocumentProcessor()
#
# @app.post("/extract")
# async def extract_text(file: UploadFile = File(...)):
#     with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
#         temp.write(await file.read())
#         temp_path = temp.name
#     try:
#         return processor.process_file(temp_path)
#     finally:
#         os.remove(temp_path)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robust OCR and document text extraction pipeline for PDFs and images."
    )
    parser.add_argument("input_file", help="Path to an input image or PDF")
    parser.add_argument("--output", "-o", default="output.json", help="Output JSON path")
    parser.add_argument(
        "--ocr", default="auto",
        choices=["auto", "paddleocr", "easyocr", "pytesseract", "tesseract"],
        help="OCR engine preference",
    )
    parser.add_argument(
        "--lang", nargs="*", default=["auto"],
        help="Language selection: auto, en, or ar",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        input_path = Path(args.input_file)
        if not input_path.is_file():
            raise FileNotFoundError(f"File not found: {args.input_file}")
        if input_path.suffix.lower() not in SUPPORTED_IMAGE_EXTS | {".pdf"}:
            raise ValueError(
                f"Unsupported file type: {input_path.suffix.lower()}. "
                "Use PDF or an image file."
            )

        processor = DocumentProcessor(ocr_backend=args.ocr, languages=args.lang)
        result = processor.process_file(str(input_path))

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(json.dumps({
            "status": result["status"],
            "page_count": result["page_count"],
            "ocr_status": result["ocr_status"],
            "output": str(out_path),
            "first_page_preview": (result["pages"][0]["text"][:500]
                                   if result["pages"] else ""),
        }, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())