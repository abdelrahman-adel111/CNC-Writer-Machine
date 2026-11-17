import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

from script1 import DocumentProcessor


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").casefold()
    text = "".join(
        char for char in text
        if unicodedata.category(char) not in {"Mn", "Cf"}
    )
    text = re.sub(r"\[page\s+\d+\]\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(left: List[str], right: List[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def error_counts(reference: str, hypothesis: str, by_word: bool = False) -> Tuple[int, int]:
    reference_text = normalize(reference)
    hypothesis_text = normalize(hypothesis)
    if by_word:
        expected = reference_text.split()
        actual = hypothesis_text.split()
    else:
        expected = list(reference_text.replace(" ", ""))
        actual = list(hypothesis_text.replace(" ", ""))
    if not expected:
        return (0 if not actual else len(actual)), 0
    return levenshtein(expected, actual), len(expected)


def error_rate(reference: str, hypothesis: str, by_word: bool = False) -> float:
    errors, denominator = error_counts(reference, hypothesis, by_word)
    if not denominator:
        return 0.0 if errors == 0 else 1.0
    return errors / denominator


def evaluate_manifest(
    manifest_path: Path, threshold: float, backend: str = "auto"
) -> Tuple[List[Dict[str, object]], float, float]:
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        samples = json.load(manifest_file)
    if not isinstance(samples, list) or not samples:
        raise ValueError("The benchmark manifest must contain a non-empty JSON list.")

    processors = {}
    results = []
    total_reference_chars = 0
    total_character_errors = 0
    total_reference_words = 0
    total_word_errors = 0

    for sample in samples:
        sample_id = sample.get("id")
        relative_image = sample.get("image")
        reference = sample.get("reference")
        if not sample_id or not relative_image or not isinstance(reference, str):
            raise ValueError(
                "Each sample requires non-empty 'id', 'image', and string 'reference' fields."
            )
        image_path = (manifest_path.parent / relative_image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Benchmark image does not exist: {image_path}")

        languages = sample.get("languages", ["en", "ar"])
        if (
            not isinstance(languages, list)
            or not languages
            or any(language not in {"en", "ar"} for language in languages)
        ):
            raise ValueError(
                f"Sample {sample_id!r} languages must be a non-empty list of 'en' and/or 'ar'."
            )
        language_key = tuple(languages)
        if language_key not in processors:
            processor = DocumentProcessor(
                ocr_backend=backend, languages=languages
            )
            if backend != "auto":
                expected_backend = (
                    "pytesseract" if backend == "tesseract" else backend
                )
                active_backend = processor._get_ocr_backend()
                if active_backend != expected_backend:
                    errors = "; ".join(
                        f"{name}: {message}"
                        for name, message in processor.backend_errors.items()
                    )
                    details = f" Details: {errors}" if errors else ""
                    raise ValueError(
                        f"Requested OCR backend {backend!r} is unavailable "
                        f"for languages {list(language_key)!r}.{details}"
                    )
            processors[language_key] = processor
        processor = processors[language_key]
        result = processor.process_file(str(image_path))
        hypothesis = result.get("text", "")
        character_errors = error_rate(reference, hypothesis)
        word_errors = error_rate(reference, hypothesis, by_word=True)
        character_error_count, reference_char_count = error_counts(reference, hypothesis)
        word_error_count, reference_word_count = error_counts(
            reference, hypothesis, by_word=True
        )
        total_reference_chars += reference_char_count
        total_character_errors += character_error_count
        total_reference_words += reference_word_count
        total_word_errors += word_error_count

        results.append({
            "id": sample_id,
            "language": sample.get("language", "unspecified"),
            "category": sample.get("category", "unspecified"),
            "cer": character_errors,
            "wer": word_errors,
            "accuracy": max(0.0, 1.0 - character_errors),
            "confidence": result.get("pages", [{}])[0].get("confidence", 0.0),
            "passed_target": character_errors <= 1.0 - threshold,
            "character_errors": character_error_count,
            "reference_characters": reference_char_count,
        })

    corpus_cer = (
        total_character_errors / total_reference_chars
        if total_reference_chars else 1.0
    )
    corpus_wer = (
        total_word_errors / total_reference_words
        if total_reference_words else 1.0
    )
    return results, corpus_cer, corpus_wer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure OCR character and word error rates against manually verified text."
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        default=str(Path(__file__).with_name("benchmark_samples.json")),
        help="JSON manifest; image paths are relative to the manifest",
    )
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument(
        "--ocr",
        choices=["auto", "paddleocr", "easyocr", "pytesseract", "tesseract"],
        default="auto",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 < args.threshold <= 1:
        print("ERROR: --threshold must be greater than 0 and at most 1.", file=sys.stderr)
        return 2
    try:
        manifest_path = Path(args.manifest).resolve()
        results, corpus_cer, corpus_wer = evaluate_manifest(
            manifest_path, args.threshold, args.ocr
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for result in results:
        print(
            f"{result['id']}: accuracy={result['accuracy']:.1%} "
            f"CER={result['cer']:.1%} WER={result['wer']:.1%} "
            f"confidence={result['confidence']:.1%} "
            f"target={'PASS' if result['passed_target'] else 'FAIL'}"
        )
    print(
        f"CORPUS: samples={len(results)} CER={corpus_cer:.1%} "
        f"WER={corpus_wer:.1%} character_accuracy={1.0 - corpus_cer:.1%} "
        f"target={args.threshold:.0%}"
    )
    for group_name in ("language", "category"):
        groups = {}
        for result in results:
            group = str(result[group_name])
            totals = groups.setdefault(
                group, {"samples": 0, "errors": 0, "characters": 0}
            )
            totals["samples"] += 1
            totals["errors"] += int(result["character_errors"])
            totals["characters"] += int(result["reference_characters"])
        for group, totals in sorted(groups.items()):
            cer = totals["errors"] / totals["characters"] if totals["characters"] else 1.0
            print(
                f"BY {group_name.upper()}: {group} samples={totals['samples']} "
                f"CER={cer:.1%} character_accuracy={max(0.0, 1.0 - cer):.1%}"
            )
    if any(not result["passed_target"] for result in results) or corpus_cer > 1.0 - args.threshold:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
