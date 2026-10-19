# Attempt log

## 2026-10-18 — renderer truncation defect found before rubric scoring

The first `cost_anchor × sentiment × rubric × choice_logprob` attempt stopped after the first batch.
The original runner applied the task's 128-token content cap to the combined system prompt, user
content, and candidate answer. The longer rubric could therefore truncate the exact `YES` or `NO`
continuation. The scorer detected a row with no continuation tokens and raised an error; no result
artifact was written.

Correction: tokenize and cap the task content first, then render the complete frozen system prompt,
chat boundary, and answer. The combined judge prompt is no longer truncated at the content cap.
The two earlier zero-shot artifacts were moved under `attempts/pre-content-cap-fix/` and are diagnostic
only. They are not eligible for the sentinel summary even though their aggregate behaviour exposed a
useful all-YES failure mode.
## 2026-10-18 — Qwen toxicity rubric choice-logprob interrupted

- Arm: `quality_anchor / toxicity / rubric / choice_logprob`
- The process stopped after 80/512 rows while other read-only repository inspection was running.
- No prediction or metadata artifact was written, so no partial result entered the summary.
- The run must restart from row zero under the unchanged frozen config and prompt.
