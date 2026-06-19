"""Entry point: process claims CSV and write output.csv."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.config import (
    DEFAULT_CLAIMS_PATH,
    DEFAULT_OUTPUT_PATH,
    GEMINI_MODEL,
    OUTPUT_COLUMNS,
)
from code.decision_engine import append_result_row, process_claims
from code.image_analyzer import ImageAnalyzer

print("MAIN MODEL:", GEMINI_MODEL)
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify visual evidence for damage claims."
    )
    parser.add_argument(
        "--claims",
        default=str(DEFAULT_CLAIMS_PATH),
        help="Path to input claims CSV",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to write output CSV",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model (default: gemini-1.5-flash — best free-tier support)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Seconds to wait after each API call (helps free-tier RPM limits)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output.csv and reprocess all rows",
    )
    args = parser.parse_args()
    load_dotenv()

    output_path = Path(args.output)
    claims_df = pd.read_csv(args.claims)

    if args.no_resume and output_path.is_file():
        output_path.unlink()

    analyzer = ImageAnalyzer(
        model_name=args.model or GEMINI_MODEL,
        request_delay=args.delay,
    )

    def on_complete(row: dict) -> None:
        append_result_row(output_path, row)

    new_results, analyzer = process_claims(
        claims_df,
        analyzer=analyzer,
        output_path=output_path if not args.no_resume else None,
        on_row_complete=on_complete,
    )

    # Merge resumed + new rows into final ordered output.
    if output_path.is_file():
        out_df = pd.read_csv(output_path)
    else:
        out_df = pd.DataFrame(new_results)

    for col in OUTPUT_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = ""

    # Reorder to match input claims order.
    order_keys = [
        f"{r['user_id']}|{r['image_paths']}" for _, r in claims_df.iterrows()
    ]
    out_df["_key"] = out_df["user_id"] + "|" + out_df["image_paths"]
    out_df = out_df.set_index("_key").reindex(order_keys).reset_index(drop=True)
    out_df = out_df[OUTPUT_COLUMNS]
    out_df.to_csv(output_path, index=False)

    print(f"Wrote {len(out_df)} rows to {output_path}")
    print(
        f"Model: {analyzer.model_name} | "
        f"calls: {analyzer.usage.model_calls} | "
        f"images: {analyzer.usage.images_processed} | "
        f"retries: {analyzer.usage.retries} | "
        f"model switches: {analyzer.usage.model_switches} | "
        f"input tokens: {analyzer.usage.input_tokens} | "
        f"output tokens: {analyzer.usage.output_tokens}"
    )


if __name__ == "__main__":
    main()
