"""Extract claimed parts and issue hints from the conversation transcript."""

import re

PART_PATTERNS = {
    "front_bumper": r"front bumper|front side|parachoques delantero",
    "rear_bumper": r"rear bumper|back bumper|parachoques trasero|parachoques de atras",
    "door": r"\bdoor\b|door panel|puerta",
    "hood": r"\bhood\b|hail",
    "windshield": r"windshield|front glass|wind screen",
    "side_mirror": r"side mirror|mirror",
    "headlight": r"headlight|head light",
    "taillight": r"taillight|tail light|back light",
    "fender": r"\bfender\b",
    "screen": r"\bscreen\b|display|pantalla",
    "keyboard": r"keyboard|keys|teclas",
    "trackpad": r"trackpad|track pad",
    "hinge": r"\bhinge\b",
    "lid": r"\blid\b",
    "corner": r"\bcorner\b",
    "port": r"\bport\b",
    "base": r"\bbase\b",
    "body": r"\bbody\b|body panel|palm-rest",
    "box": r"\bbox\b|cardboard box|shipping box|delivery box",
    "package_corner": r"package corner|box corner|corner crushed|corner dab",
    "package_side": r"package side|package surface|wet box",
    "seal": r"\bseal\b|torn.open|torn open|tape broken",
    "label": r"\blabel\b|unreadable label",
    "contents": r"missing contents|contents missing|product inside|item missing",
    "item": r"item inside|inside item|broken item|product inside",
}

ISSUE_PATTERNS = {
    "dent": r"\bdent\b|dented|dab gaya",
    "scratch": r"\bscratch\b|scrape|scratched",
    "crack": r"\bcrack\b|cracked|shatter",
    "glass_shatter": r"shatter|shattered",
    "broken_part": r"broken|broke|toot gaya",
    "missing_part": r"missing key|key missing|keycap",
    "torn_packaging": r"torn|torn.open|torn open|phati hui",
    "crushed_packaging": r"crush|crushed",
    "water_damage": r"water damage|wet|liquid damage",
    "stain": r"\bstain\b|oil stain|oily mark",
}


def _find_first(patterns: dict[str, str], text: str) -> str | None:
    lowered = text.lower()
    for label, pattern in patterns.items():
        if re.search(pattern, lowered, re.IGNORECASE):
            return label
    return None


def parse_claim(user_claim: str) -> dict[str, str | list[str]]:
    """Return lightweight NLP hints from the chat transcript."""
    claimed_parts: list[str] = []
    for part, pattern in PART_PATTERNS.items():
        if re.search(pattern, user_claim, re.IGNORECASE):
            claimed_parts.append(part)

    issue_hint = _find_first(ISSUE_PATTERNS, user_claim)
    return {
        "claimed_parts": claimed_parts,
        "issue_hint": issue_hint or "unknown",
        "has_instruction_injection": bool(
            re.search(
                r"approve (the )?claim|skip manual review|ignore (all )?previous|follow (the )?note|mark this row",
                user_claim,
                re.IGNORECASE,
            )
        ),
    }
