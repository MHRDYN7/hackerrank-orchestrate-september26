# Handoff for the next agent

Deadline: **2026-09-13 18:00 IST**. If that has passed, still finish `output.csv`.

This session could not see `GOOGLE_API_KEY_2` or `GOOGLE_API_KEY_3` (only the original `GOOGLE_API_KEY` is in the cloud env). Those two keys were added on the user side after 3.5 had **446** calls and 3.1 had **502** (unusable) on the first key.

## What you must do

1. Confirm keys without printing secrets:
   `GOOGLE_API_KEY_2` and `GOOGLE_API_KEY_3` set, plus maybe `GOOGLE_API_KEY`.
   `collect_keys()` already reads `GOOGLE_API_KEY_2..7` **first**, then the base key (exhausted).
2. Unset leftover `GEMINI_MODEL` so fresh keys start on `gemini-3.5-flash-lite`. Never use Pro/non-lite.
3. **Agent 25 first** (explanations matter; do not `--engine-only`):

```bash
cd code
unset GEMINI_MODEL
export GEMINI_MAX_INFLIGHT=10
uv run python main.py --samples --concurrency 10
```

4. Compare `code/evaluation/sample_predictions.csv` to `dataset/sample_requests.csv`. Check method/status/plan and that `decision_explanation` is two sentences (action, amounts/dates/cuts, then minimum balance). The last agent 25 (same prompt) already looked good except **06** (should cut-and-pay today, not refuse), **13** (should wait, not refuse), **21** (pay today but gold also stops/reduces two subscriptions).
5. If that 25 is acceptable, run the **250-row eval** immediately:

```bash
cd code
unset GEMINI_MODEL
export GEMINI_MAX_INFLIGHT=10
uv run python main.py --concurrency 10
```

Writes `/workspace/output.csv` and `code/evaluation/usage_report.md`. Fill the usage report with the **final 250-run** providers, model names, calls, input/output tokens, totals, averages, estimated cost. No API keys in that file.

6. Commit, push branch `cursor/sample-agent-iteration-9b40`, update the existing PR. Never commit `.env`.
7. Submission URL if asked: https://www.hackerrank.com/contests/hackerrank-orchestrate-september26/challenges/buy-or-wait/submission

## Architecture (do not revert)

- First human message is raw `request_text`. Ids live in the system prompt.
- `get_context` includes `spec_ranked`. `commit_decision` records the spec ranking (method/status/plan/amount/earliest/cuts) and keeps the model explanation when the method matches.
- Ranking: only plans that finish by the deadline; installment/partial/full beat wait; `not_recommended` if nothing completes in-window.
- Round-robin live keys. Per-key 3.5 → 3.1 then retire that key. RPM 429 on one key should hop to another with no long sleep.
- LangSmith project must stay `hackerrank`. `thinking_level=high`. Flash-Lite only.
- Log to `/workspace/log.txt` with `tool=Cursor` (or this harness’s exact name). Append only.

## Do not

- Call Gemini on the exhausted first key in a tight 429 loop.
- Run engine-only for the scored submission unless every key is dead.
- Overfit the system prompt to public sample ids.
