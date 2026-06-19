"""Combine history, evidence rules, and vision analysis into a final row."""

from pathlib import Path
from typing import Any, Callable

import pandas as pd
from tqdm import tqdm

from code.claim_parser import parse_claim
from code.config import OUTPUT_COLUMNS
from code.evidence_checker import get_relevant_requirements, load_evidence_requirements
from code.history_checker import get_history_context, load_history
from code.image_analyzer import ImageAnalyzer


def _format_row(row: dict[str, Any]) -> dict[str, Any]:
    row["evidence_standard_met"] = (
        "true" if str(row.get("evidence_standard_met")).lower() in ("true", "1") else "false"
    )
    row["valid_image"] = (
        "true" if str(row.get("valid_image")).lower() in ("true", "1") else "false"
    )
    return row


def _claim_key(row: pd.Series) -> str:
    return f"{row['user_id']}|{row['image_paths']}"


def _load_completed_keys(output_path: Path) -> set[str]:
    if not output_path.is_file():
        return set()
    try:
        done = pd.read_csv(output_path)
        if done.empty:
            return set()
        return {
            f"{r['user_id']}|{r['image_paths']}"
            for _, r in done.iterrows()
        }
    except Exception:
        return set()


def process_claim_row(
    row: pd.Series,
    history_df: pd.DataFrame,
    requirements_df: pd.DataFrame,
    analyzer: ImageAnalyzer,
) -> dict[str, Any]:
    user_id = str(row["user_id"])
    image_paths = str(row["image_paths"])
    user_claim = str(row["user_claim"])
    claim_object = str(row["claim_object"])

    history_context = get_history_context(user_id, history_df)
    claim_hints = parse_claim(user_claim)
    evidence_rules = get_relevant_requirements(
        claim_object,
        claim_hints["claimed_parts"],
        str(claim_hints["issue_hint"]),
        requirements_df,
    )

    analysis = analyzer.analyze(
        user_claim=user_claim,
        claim_object=claim_object,
        image_paths=image_paths,
        history_context=history_context,
        evidence_rules=evidence_rules,
        claim_hints=claim_hints,
    )

    if claim_hints.get("has_instruction_injection"):
        flags = analysis["risk_flags"]
        if flags == "none":
            analysis["risk_flags"] = "text_instruction_present"
        elif "text_instruction_present" not in flags:
            analysis["risk_flags"] = flags + ";text_instruction_present"

    return _format_row(
        {
            "user_id": user_id,
            "image_paths": image_paths,
            "user_claim": user_claim,
            "claim_object": claim_object,
            **analysis,
        }
    )


def process_claims(
    claims_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    requirements_df: pd.DataFrame | None = None,
    analyzer: ImageAnalyzer | None = None,
    output_path: Path | None = None,
    on_row_complete: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], ImageAnalyzer]:
    history_df = history_df or load_history()
    requirements_df = requirements_df or load_evidence_requirements()
    analyzer = analyzer or ImageAnalyzer()

    completed_keys: set[str] = set()
    if output_path:
        completed_keys = _load_completed_keys(output_path)

    results: list[dict[str, Any]] = []
    pending = [
        row for _, row in claims_df.iterrows()
        if _claim_key(row) not in completed_keys
    ]

    if completed_keys:
        print(f"Skipping {len(completed_keys)} already-completed rows.")

    for row in tqdm(pending, total=len(pending), desc="Claims"):
        result = process_claim_row(row, history_df, requirements_df, analyzer)
        results.append(result)
        if on_row_complete:
            on_row_complete(result)

    return results, analyzer


def append_result_row(output_path: Path, row: dict[str, Any]) -> None:
    """Append one formatted row to output CSV (creates file with header if needed)."""
    out = {col: row.get(col, "") for col in OUTPUT_COLUMNS}
    df = pd.DataFrame([out])
    write_header = not output_path.is_file() or output_path.stat().st_size == 0
    df.to_csv(output_path, mode="a", header=write_header, index=False)
