# Buy or Wait?

Deterministic cash-flow engine plus a thin LangGraph loop (`gemini-3.5-flash-lite`). The engine owns every number; the model picks among scored candidates and writes the explanation.

## Setup

From this `code/` directory:

```bash
uv sync
```

Copy `.env.example` to `.env` (repo root or `code/`) and set `GOOGLE_API_KEY` plus `GOOGLE_API_KEY_2` / `GOOGLE_API_KEY_3` if you have extra Flash-Lite keys. Numbered keys are used first. The run starts on `gemini-3.5-flash-lite` at `thinking_level=high` and `GEMINI_MAX_INFLIGHT=10`. After a daily RPD 429 a key switches to `gemini-3.1-flash-lite`, then that key is retired. Only Flash-Lite models are used. Without a live key the run cannot score explanations.

LangSmith traces are always written to project `hackerrank` (`LANGSMITH_PROJECT` is forced in code).

## Run

```bash
unset GEMINI_MODEL
export GEMINI_MAX_INFLIGHT=10

# 25 public samples (agent + explanations)
uv run python main.py --samples --concurrency 10

# Full evaluation set → ../output.csv
uv run python main.py --concurrency 10
```

Reads only `../dataset/`. Writes `../output.csv` and `evaluation/usage_report.md`.
