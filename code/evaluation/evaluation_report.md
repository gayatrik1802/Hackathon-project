# Evaluation Report

Run `python code/evaluation/main.py` after setting `GEMINI_API_KEY` to generate metrics and update this report.

## Sample set metrics (`dataset/sample_claims.csv`)

Pending — evaluation not yet run.

## Final strategy for `output.csv`

- Model: `gemini-2.0-flash` (override with `GEMINI_MODEL`)
- Single multimodal call per claim row with structured JSON output
- History flags merged after vision analysis; visuals remain primary

## Operational analysis (test set estimates)

- **Model calls**: ~1 per claim row (~45 rows in `claims.csv`)
- **Images processed**: typically 1–3 per row
- **Cost estimate**: roughly $0.01–$0.02 for full test set on Gemini 2.0 Flash
- **Runtime**: ~2–4s per row sequential → ~2–3 minutes total
- **Rate limits**: sequential calls with exponential backoff retries
