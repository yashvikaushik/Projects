"""
main.py
-------
CLI entry point for the Medical Symptom Extraction Pipeline.

Usage
-----
  # Interactive mode
  python main.py

  # Single-shot mode
  python main.py --text "I have a sharp pain in my lower back for 2 days. No fever."

  # Pipe mode
  echo "severe headache since yesterday, no nausea" | python main.py --stdin
"""

from __future__ import annotations

import argparse
import json
import sys

from nlp_engine import NLPEngine
from symptom_extractor import SymptomExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medical_symptom_pipeline",
        description="Extract structured symptom data from unstructured medical text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text string to process (single-shot mode).",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read input text from stdin.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model name (default: en_core_web_sm). "
             "Use 'en_core_sci_sm' for SciSpacy.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True).",
    )
    return parser


def run_pipeline(text: str, engine: NLPEngine, extractor: SymptomExtractor) -> dict:
    """
    Core pipeline:  raw text → NLP features → structured symptoms.
    Returns the JSON-serializable result dict.
    """
    nlp_features = engine.process(text)
    result = extractor.extract(nlp_features, original_text=text)
    return result.model_dump()


def print_result(result: dict, pretty: bool = True) -> None:
    indent = 2 if pretty else None
    print(json.dumps(result, indent=indent, ensure_ascii=False))


def interactive_loop(engine: NLPEngine, extractor: SymptomExtractor) -> None:
    print("Medical Symptom Extraction Pipeline")
    print("Enter symptom description (or 'quit' to exit):")
    print("-" * 50)

    while True:
        try:
            text = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit", "q"}:
            break

        result = run_pipeline(text, engine, extractor)
        print_result(result)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Initialise pipeline components once (model loading is expensive)
    engine = NLPEngine(model_name=args.model)
    extractor = SymptomExtractor()

    if args.stdin:
        text = sys.stdin.read().strip()
        if not text:
            print("Error: no input received from stdin.", file=sys.stderr)
            sys.exit(1)
        result = run_pipeline(text, engine, extractor)
        print_result(result, args.pretty)

    elif args.text:
        result = run_pipeline(args.text, engine, extractor)
        print_result(result, args.pretty)

    else:
        interactive_loop(engine, extractor)


if __name__ == "__main__":
    main()
