"""Shared helpers for paths, parsing, and output normalization."""

import re
from pathlib import Path
from typing import Any

from code.config import (
    CLAIM_STATUSES,
    ISSUE_TYPES,
    PARTS_BY_OBJECT,
    RISK_FLAGS,
    SEVERITIES,
    ROOT,
)


def resolve_image_path(path_str: str) -> Path:
    """Resolve an image path relative to dataset/ or project root."""
    path = Path(path_str)
    if path.is_file():
        return path
    dataset_path = ROOT / "dataset" / path_str
    if dataset_path.is_file():
        return dataset_path
    root_path = ROOT / path_str
    if root_path.is_file():
        return root_path
    return dataset_path


def parse_image_paths(image_paths: str) -> list[tuple[str, Path]]:
    """Return (image_id, resolved_path) pairs from a semicolon-separated path string."""
    pairs: list[tuple[str, Path]] = []
    for raw in str(image_paths).split(";"):
        raw = raw.strip()
        if not raw:
            continue
        image_id = Path(raw).stem
        pairs.append((image_id, resolve_image_path(raw)))
    return pairs


def normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("true", "1", "yes")


def _pick_allowed(value: str, allowed: tuple[str, ...], fallback: str) -> str:
    cleaned = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned in allowed:
        return cleaned
    for item in allowed:
        if item in cleaned or cleaned in item:
            return item
    return fallback


def normalize_risk_flags(flags: Any, history_flags: list[str]) -> str:
    collected: set[str] = set()
    if isinstance(flags, list):
        for f in flags:
            if f and str(f).strip().lower() != "none":
                collected.add(_pick_allowed(str(f), RISK_FLAGS, str(f)))
    elif isinstance(flags, str):
        for part in flags.split(";"):
            part = part.strip()
            if part and part.lower() != "none":
                collected.add(_pick_allowed(part, RISK_FLAGS, part))
    for hf in history_flags:
        if hf and hf.lower() != "none":
            collected.add(_pick_allowed(hf, RISK_FLAGS, hf))
    if not collected:
        return "none"
    ordered = [f for f in RISK_FLAGS if f in collected]
    extras = sorted(collected - set(ordered))
    return ";".join(ordered + extras)


def normalize_supporting_ids(value: Any) -> str:
    if value is None or str(value).strip().lower() in ("", "none", "null"):
        return "none"
    if isinstance(value, list):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return ";".join(parts) if parts else "none"
    return str(value).strip()


def normalize_analysis(
    raw: dict[str, Any],
    claim_object: str,
    history_flags: list[str],
) -> dict[str, Any]:
    """Normalize model output to allowed schema values."""
    parts = PARTS_BY_OBJECT.get(claim_object, ("unknown",))
    return {
        "evidence_standard_met": normalize_bool(raw.get("evidence_standard_met")),
        "evidence_standard_met_reason": str(
            raw.get("evidence_standard_met_reason", "Unable to evaluate evidence.")
        ).strip(),
        "risk_flags": normalize_risk_flags(raw.get("risk_flags"), history_flags),
        "issue_type": _pick_allowed(
            str(raw.get("issue_type", "unknown")), ISSUE_TYPES, "unknown"
        ),
        "object_part": _pick_allowed(
            str(raw.get("object_part", "unknown")), parts, "unknown"
        ),
        "claim_status": _pick_allowed(
            str(raw.get("claim_status", "not_enough_information")),
            CLAIM_STATUSES,
            "not_enough_information",
        ),
        "claim_status_justification": str(
            raw.get("claim_status_justification", "Insufficient visual evidence.")
        ).strip(),
        "supporting_image_ids": normalize_supporting_ids(
            raw.get("supporting_image_ids")
        ),
        "valid_image": normalize_bool(raw.get("valid_image"), default=True),
        "severity": _pick_allowed(
            str(raw.get("severity", "unknown")), SEVERITIES, "unknown"
        ),
    }


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from model text, including fenced code blocks."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")
    return __import__("json").loads(text[start:end + 1])
