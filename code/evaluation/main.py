"""Evaluate predictions against sample_claims.csv and compare model strategies."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.config import DATASET_DIR, GEMINI_MODEL, OUTPUT_COLUMNS
from code.decision_engine import process_claims
from code.image_analyzer import ImageAnalyzer

SAMPLE_PATH = DATASET_DIR / "sample_claims.csv"
EVAL_DIR = Path(__file__).resolve().parent


def _normalize_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna("").astype(str).str.strip().str.lower()


def score_predictions(expected: pd.DataFrame, predicted: pd.DataFrame) -> dict:
    metrics: dict[str, float] = {}
    scored_cols = [
        "evidence_standard_met",
        "claim_status",
        "issue_type",
        "object_part",
        "severity",
        "valid_image",
        "risk_flags",
    ]
    for col in scored_cols:
        exp = _normalize_series(expected, col)
        pred = _normalize_series(predicted, col)
        metrics[col] = float((exp == pred).mean())
    metrics["exact_row_match"] = float(
        (_normalize_series(expected, "claim_status") == _normalize_series(predicted, "claim_status"))
        & (_normalize_series(expected, "issue_type") == _normalize_series(predicted, "issue_type"))
        & (_normalize_series(expected, "object_part") == _normalize_series(predicted, "object_part"))
    ).mean()
    return metrics


def run_strategy(
    sample_df: pd.DataFrame,
    model_name: str,
    input_cols: list[str],
) -> tuple[pd.DataFrame, ImageAnalyzer]:
    input_df = sample_df[input_cols].copy()
    analyzer = ImageAnalyzer(model_name=model_name)
    results, analyzer = process_claims(input_df, analyzer=analyzer)
    pred_df = pd.DataFrame(results)
    for col in OUTPUT_COLUMNS:
        if col not in pred_df.columns:
            pred_df[col] = ""
    return pred_df[OUTPUT_COLUMNS], analyzer


def format_report(
    strategy_results: list[tuple[str, dict, ImageAnalyzer]],
    final_model: str,
) -> str:
    lines = [
        "# Evaluation Report",
        "",
        "## Sample set metrics (`dataset/sample_claims.csv`)",
        "",
    ]
    for name, metrics, analyzer in strategy_results:
        lines.append(f"### Strategy: {name}")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value:.1%}")
        lines.append(
            f"- model_calls: {analyzer.usage.model_calls}, "
            f"images: {analyzer.usage.images_processed}, "
            f"input_tokens: {analyzer.usage.input_tokens}, "
            f"output_tokens: {analyzer.usage.output_tokens}, "
            f"retries: {analyzer.usage.retries}"
        )
        lines.append("")

    lines.extend(
        [
            "## Final strategy for `output.csv`",
            f"- Model: `{final_model}`",
            "- Single multimodal call per claim row with structured JSON output",
            "- History flags merged after vision analysis; visuals remain primary",
            "",
            "## Operational analysis (test set estimates)",
            "",
            "- **Model calls**: 1 per claim row (~45 rows in `claims.csv`) = ~45 calls",
            "- **Images processed**: sum of images per row (typically 1–3) ≈ 80–100 images",
            "- **Token usage**: ~1,500–2,500 input tokens and ~200–400 output tokens per row "
            "(prompt + 1–3 images); full test set ≈ 70k–110k input tokens",
            "- **Cost estimate** (Gemini 2.0 Flash): ~$0.10/1M input tokens, ~$0.40/1M output "
            "→ roughly **$0.01–$0.02** for the full test set",
            "- **Runtime**: ~2–4s per row with sequential calls → ~2–3 minutes total",
            "- **Rate limits**: Sequential processing with exponential backoff retries; "
            "no duplicate calls per row; history/evidence loaded once per batch",
            "",
            "## Notes",
            "- Two strategies compared above (primary vs alternate Gemini model or temperature).",
            "- Instruction injection in chat/images is flagged via `text_instruction_present`.",
            "- User history adds `user_history_risk` / `manual_review_required` without overriding visuals.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate claim verification system.")
    parser.add_argument(
        "--sample",
        default=str(SAMPLE_PATH),
        help="Path to labeled sample CSV",
    )
    parser.add_argument(
        "--strategy-a",
        default=GEMINI_MODEL,
        help="Primary Gemini model (default: gemini-1.5-flash)",
    )
    parser.add_argument(
        "--strategy-b",
        default="gemini-1.5-flash",
        help="Alternate Gemini model for comparison",
    )
    parser.add_argument(
        "--final-model",
        default=GEMINI_MODEL,
        help="Model used for final predictions",
    )
    parser.add_argument(
        "--report",
        default=str(EVAL_DIR / "evaluation_report.md"),
        help="Where to write evaluation_report.md",
    )
    args = parser.parse_args()

    sample_df = pd.read_csv(args.sample)
    input_cols = ["user_id", "image_paths", "user_claim", "claim_object"]
    expected = sample_df[OUTPUT_COLUMNS].copy()

    strategy_results: list[tuple[str, dict, ImageAnalyzer]] = []
    for label, model in [
        ("A: " + args.strategy_a, args.strategy_a),
        ("B: " + args.strategy_b, args.strategy_b),
    ]:
        print(f"Running strategy {label}...")
        predicted, analyzer = run_strategy(sample_df, model, input_cols)
        metrics = score_predictions(expected, predicted)
        strategy_results.append((label, metrics, analyzer))
        print(f"  claim_status accuracy: {metrics['claim_status']:.1%}")

    report = format_report(strategy_results, args.final_model)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
