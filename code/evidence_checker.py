"""Select relevant minimum evidence requirements for a claim."""

import pandas as pd

from code.config import DEFAULT_EVIDENCE_PATH


def load_evidence_requirements(path: str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DEFAULT_EVIDENCE_PATH)


def get_relevant_requirements(
    claim_object: str,
    claimed_parts: list[str],
    issue_hint: str,
    requirements_df: pd.DataFrame,
) -> list[str]:
    """Return human-readable evidence rules applicable to this claim."""
    selected: list[str] = []
    for _, row in requirements_df.iterrows():
        obj = str(row["claim_object"])
        if obj != "all" and obj != claim_object:
            continue
        applies = str(row["applies_to"]).lower()
        minimum = str(row["minimum_image_evidence"])
        if obj == "all":
            selected.append(f"[{row['requirement_id']}] {minimum}")
            continue

        part_match = any(part.replace("_", " ") in applies or part in applies for part in claimed_parts)
        issue_match = issue_hint != "unknown" and issue_hint.replace("_", " ") in applies
        keyword_match = any(
            kw in applies
            for kw in ("general", "review", "multi-image", "contents", "inner")
        )
        if part_match or issue_match or keyword_match or not claimed_parts:
            selected.append(f"[{row['requirement_id']}] {minimum}")

    if not selected:
        for _, row in requirements_df.iterrows():
            if str(row["claim_object"]) in ("all", claim_object):
                selected.append(
                    f"[{row['requirement_id']}] {row['minimum_image_evidence']}"
                )
    return selected
