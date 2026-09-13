# Buy or Wait?

Deterministic cash-flow engine plus a thin LangGraph loop (`gemini-3.5-flash-lite`). The engine owns every number; the model picks among scored candidates and writes the explanation.

## Setup

From this `code/` directory:

```bash
uv sync
```

Copy `../.env.example` to `../.env` and add numbered Gemini keys (`GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, …). Keys are optional: without them the run is engine-only (template explanations).

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
