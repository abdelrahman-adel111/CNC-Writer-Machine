import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2

from evaluate_ocr import error_counts
from script1 import DocumentProcessor


AVAILABLE_ENGINES = ("paddleocr", "easyocr", "pytesseract")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare OCR engines on identical original document images, "
            "without engine-specific preprocessing."
        )
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        default=str(Path(__file__).with_name("benchmark_document_samples.json")),
        help="JSON manifest; image paths are relative to the manifest",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=AVAILABLE_ENGINES,
        default=["paddleocr", "easyocr", "pytesseract"],
    )
    parser.add_argument(
        "--source-rows",
        nargs="*",
        type=int,
        help="Optional source row indices to evaluate",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("benchmark_document_engine_results.json")),
        help="Path for the machine-readable comparison report",
    )
    return parser


def corpus_rate(results: List[Dict[str, object]], metric: str) -> float:
    if metric == "character":
        error_key = "character_errors"
        reference_key = "reference_characters"
    else:
        error_key = "word_errors"
        reference_key = "reference_words"
    errors = sum(int(result[error_key]) for result in results)
    denominator = sum(int(result[reference_key]) for result in results)
    return errors / denominator if denominator else 0.0


def compare_engines(
    manifest_path: Path, engines: List[str], source_rows: List[int] = None
) -> Dict[str, object]:
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        samples = json.load(manifest_file)
    if not isinstance(samples, list) or not samples:
        raise ValueError("The benchmark manifest must contain a non-empty JSON list.")

    if source_rows is not None:
        wanted = set(source_rows)
        samples = [
            sample for sample in samples
            if sample.get("source", {}).get("row") in wanted
        ]
        found = {sample.get("source", {}).get("row") for sample in samples}
        missing = wanted - found
        if missing:
            raise ValueError(f"Source rows are not in the manifest: {sorted(missing)}")

    processors = {}
    reports = {}
    for engine in engines:
        engine_results = []
        engine_start = time.perf_counter()
        for sample in samples:
            relative_image = sample.get("image")
            reference = sample.get("reference")
            if not relative_image or not isinstance(reference, str) or not reference:
                raise ValueError(
                    "Each sample requires non-empty 'image' and string 'reference' fields."
                )
            image_path = (manifest_path.parent / relative_image).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Benchmark image does not exist: {image_path}")

            languages = sample.get("languages", ["ar"])
            if (
                not isinstance(languages, list)
                or not languages
                or any(language not in {"en", "ar"} for language in languages)
            ):
                raise ValueError(
                    f"Sample {sample.get('id')!r} languages must contain 'en' and/or 'ar'."
                )
            processor_key = (engine, tuple(languages))
            if processor_key not in processors:
                processors[processor_key] = DocumentProcessor(
                    ocr_backend=engine, languages=languages
                )
            processor = processors[processor_key]
            if processor._get_ocr_backend() != engine:
                engine_results.append({
                    "id": sample.get("id"),
                    "source_row": sample.get("source", {}).get("row"),
                    "status": "unavailable",
                    "reason": processor.backend_errors.get(
                        engine, f"{engine} is not available in this environment."
                    ),
                })
                continue

            image = processor.read_image(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            start = time.perf_counter()
            text, confidence = getattr(processor, f"ocr_{engine}")(image)
            elapsed = time.perf_counter() - start
            char_errors, reference_chars = error_counts(reference, text)
            word_errors, reference_words = error_counts(reference, text, by_word=True)
            engine_results.append({
                "id": sample.get("id"),
                "source_row": sample.get("source", {}).get("row"),
                "status": "ok",
                "language": sample.get("language", "unspecified"),
                "category": sample.get("category", "unspecified"),
                "cer": char_errors / reference_chars if reference_chars else 0.0,
                "wer": word_errors / reference_words if reference_words else 0.0,
                "character_errors": char_errors,
                "reference_characters": reference_chars,
                "word_errors": word_errors,
                "reference_words": reference_words,
                "confidence": confidence,
                "seconds": elapsed,
            })

        successful = [item for item in engine_results if item["status"] == "ok"]
        reports[engine] = {
            "samples": len(engine_results),
            "successful_samples": len(successful),
            "unavailable_samples": len(engine_results) - len(successful),
            "cer": corpus_rate(successful, "character"),
            "wer": corpus_rate(successful, "word"),
            "character_accuracy": max(0.0, 1.0 - corpus_rate(successful, "character")),
            "elapsed_seconds": time.perf_counter() - engine_start,
            "results": engine_results,
        }

    return {
        "manifest": manifest_path.name,
        "method": "one inference pass per engine on the same original RGB image; no OCR preprocessing variants",
        "sample_count": len(samples),
        "engines": reports,
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest_path = Path(args.manifest).resolve()
        report = compare_engines(manifest_path, args.engines, args.source_rows)
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, ensure_ascii=False, indent=2)
            report_file.write("\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for engine, summary in report["engines"].items():
        if summary["successful_samples"]:
            print(
                f"{engine}: samples={summary['successful_samples']} "
                f"CER={summary['cer']:.1%} WER={summary['wer']:.1%} "
                f"character_accuracy={summary['character_accuracy']:.1%} "
                f"elapsed={summary['elapsed_seconds']:.1f}s"
            )
        else:
            print(f"{engine}: unavailable")
        for result in summary["results"]:
            if result["status"] == "ok":
                print(
                    f"  row {result['source_row']}: CER={result['cer']:.1%} "
                    f"WER={result['wer']:.1%} confidence={result['confidence']:.1%}"
                )
            else:
                print(f"  row {result['source_row']}: unavailable ({result['reason']})")
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
