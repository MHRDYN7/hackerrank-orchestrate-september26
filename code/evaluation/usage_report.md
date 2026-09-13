# Token usage report

Generated: 2026-09-13T12:15:12.125547+00:00

## Models

- Provider: Google
- Models: `gemini-3.5-flash-lite` (primary), then `gemini-3.1-flash-lite` after per-key daily/RPM 429s
- Thinking: `thinking_level=high`
- LangSmith project: `hackerrank`

## Final full-dataset run

- Requests: 250
- Model calls: 1266
- Input tokens: 9638768
- Output tokens: 409178
- Total tokens: 10047946
- Average tokens per request: 40191.78
- Estimated total cost (USD): 1.004795 (placeholder rate USD 0.10 / 1M tokens; Flash-Lite list prices are lower)
- Estimated cost per request (USD): 0.004019

This was an agent run (not engine-only). Numeric fields come from the cash engine; the model wrote `decision_explanation` after `commit_decision`.
