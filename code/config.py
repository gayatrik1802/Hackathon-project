"""Configuration, paths, and allowed output values."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
DEFAULT_CLAIMS_PATH = DATASET_DIR / "claims.csv"
DEFAULT_OUTPUT_PATH = ROOT / "output.csv"
DEFAULT_HISTORY_PATH = DATASET_DIR / "user_history.csv"
DEFAULT_EVIDENCE_PATH = DATASET_DIR / "evidence_requirements.csv"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Free-tier friendly fallbacks when primary model quota is exhausted.
GEMINI_FALLBACK_MODELS = tuple(
    m.strip()
    for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.5-flash,gemini-2.5-flash-lite"
    ).split(",")
    if m.strip()
)
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_STATUSES = ("supported", "contradicted", "not_enough_information")
SEVERITIES = ("none", "low", "medium", "high", "unknown")

ISSUE_TYPES = (
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
)

CAR_PARTS = (
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "unknown",
)

LAPTOP_PARTS = (
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "body",
    "unknown",
)

PACKAGE_PARTS = (
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
)

RISK_FLAGS = (
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
)

PARTS_BY_OBJECT = {
    "car": CAR_PARTS,
    "laptop": LAPTOP_PARTS,
    "package": PACKAGE_PARTS,
}
