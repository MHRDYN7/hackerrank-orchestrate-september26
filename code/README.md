# Buy or Wait?

Deterministic cash-flow engine plus a thin LangGraph loop (`gemini-3.5-flash-lite`). The engine owns every number; the model picks among scored candidates and writes the explanation.

## Setup

From this `code/` directory:

```bash
uv sync
```

Copy `.env.example` to `.env` (repo root or `code/`) and set `GEMINI_API_KEY`. The run starts on `gemini-3.5-flash-lite` at `thinking_level=high`. After a 429 / daily RPD cap it switches to `gemini-3.1-flash-lite` on the same key. Only Flash-Lite models are used. Without a key the run is engine-only.

## Run

```bash
# Full evaluation set → ../output.csv
uv run python main.py

# Score the 25 public samples (no LLM required)
uv run python main.py --samples

# Rebuild the derived SQLite cache
uv run python main.py --preprocess-only
```

Reads only `../dataset/`. Writes `../output.csv` and `evaluation/usage_report.md`.
