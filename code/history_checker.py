"""Derive risk flags from user claim history."""

import pandas as pd

from code.config import DEFAULT_HISTORY_PATH


def load_history(path: str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DEFAULT_HISTORY_PATH)


def get_history_context(user_id: str, history_df: pd.DataFrame) -> dict:
    """Return history summary and risk flags for a user."""
    row = history_df[history_df["user_id"] == user_id]
    if row.empty:
        return {
            "history_flags": [],
            "history_summary": "No prior claim history found.",
            "last_90_days_claim_count": 0,
            "manual_review_claim": 0,
            "rejected_claim": 0,
        }

    record = row.iloc[0]
    flags: list[str] = []
    raw_flags = str(record.get("history_flags", "none"))
    if raw_flags and raw_flags.lower() != "none":
        flags.extend([f.strip() for f in raw_flags.split(";") if f.strip()])

    last_90 = int(record.get("last_90_days_claim_count", 0))
    manual_review = int(record.get("manual_review_claim", 0))
    rejected = int(record.get("rejected_claim", 0))

    if last_90 > 5 or rejected > 2:
        flags.append("user_history_risk")
    if manual_review > 2 or "manual_review_required" in raw_flags:
        flags.append("manual_review_required")

    return {
        "history_flags": list(dict.fromkeys(flags)),
        "history_summary": str(record.get("history_summary", "")),
        "last_90_days_claim_count": last_90,
        "manual_review_claim": manual_review,
        "rejected_claim": rejected,
    }
