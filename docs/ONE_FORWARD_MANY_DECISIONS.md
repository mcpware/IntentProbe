# IntentProbe: One Forward, Many Decisions

**Public architecture note v0.1.0**<br>
**Author:** Nicole BeWater<br>
**Publisher:** MCPware, a product and research brand of GEASS LABS INC.<br>
**Published:** 18 October 2026

A language model can carry a useful pattern in its hidden states without turning that pattern into a
reliable `YES` or `NO` answer. IntentProbe reads the pattern directly.

![One input goes through one frozen local model. The same hidden states feed many small task heads,
which produce calibrated scores and a runtime receipt.](one-forward-many-decisions.svg)

## The short version

Think of the local language model as a brain and each activation head as a small meter attached to
it. The model reads an input once. While it reads, its hidden layers react differently to toxicity,
sentiment, spam, jailbreak attempts, hostility and other concepts. A separately trained head reads
the part of that reaction that matters for one task.

The heads do not ask the model to explain itself. They turn hidden-state patterns into scores using
decision boundaries learned from labelled examples.

That gives the runtime a simple shape:

1. render the request and context once;
2. run one frozen local model forward pass;
3. expose the required hidden states through an activation bus;
4. apply many small, independently versioned heads;
5. calibrate the scores and pass them to host policy;
6. record what ran in a receipt.

The expensive part is the model forward pass. The heads are small enough that adding more of them
costs very little.

## The model may know more than it can say

Knowing something and producing the requested answer are different interfaces.

When a direct LLM judge is asked to answer `YES` or `NO`, the model has to understand the prompt,
match the writer's label definition, map its internal representation to two output tokens, follow
the response format and avoid adding anything else. The wording of the prompt can change the result.

An activation head takes a shorter path. It reads the model's internal response before generation
and learns the task boundary from labelled examples. A prompt can describe a rule in words. A
trained head can compress thousands of examples of that rule into a small set of weights.

This does not mean an activation head must always be more accurate. It means that a useful signal can
already exist inside the model even when prompted output is a poor way to recover it.

Our direct comparison shows all three possible outcomes:

| Same-backbone comparison | What happened |
|---|---|
| Toxicity | The trained activation heads beat both tested direct-judge prompts on both local backbones. All four corrected comparisons had `q=0.0024`. |
| Sentiment on Qwen3.5-0.8B | Activation reached `0.9424` AUPRC. The best tested direct judge reached `0.9487`. The difference was not statistically distinguishable (`q=0.7971`). |
| Authorization on Qwen3.5-0.8B | The current activation head reached `0.3567`; the zero-shot choice judge reached `0.5704`. The judge was better in this cell. |

This is why the architecture has a quality gate. A cheap head is useful only when its held-out
quality, calibration and shift checks are good enough for its intended job.

## What the first systems benchmark found

We froze 16 independently trained binary heads on one exact Qwen2.5-0.5B revision. The suite covers
toxicity, jailbreak attempts, spam, positive sentiment, five emotion labels and seven conversational
pragmatics labels.

On the recorded six-core CPU, batch-one mean latency was:

| Runtime path | Mean latency per input |
|---|---:|
| One model forward + 1 head | 68.63 ms |
| One model forward + 16 heads | 73.34 ms |
| 16 repeated base-model forwards | 956.71 ms |

Going from one head to sixteen increased mean latency by 6.9%: ratio `1.069`, paired-bootstrap 95%
CI `[1.056, 1.078]`. On the matched ten-prompt comparison, 16 repeated forwards were 15.26 times
slower than the shared path, 95% CI `[15.04, 15.49]`.

The repeated-forward measurement is deliberately conservative. It omits the task prompt,
language-model output head, label scoring, parsing and generation required by a full LLM judge.

The 16 heads themselves took 1.85 ms per input. Peak process memory at 16 heads was only 0.73 MiB
above the one-head run, and the saved probe artifacts occupied 386,121 bytes.

![Measured latency stays nearly flat as heads are added, while repeated model forwards rise almost
linearly.](one-forward-many-decisions-latency.svg)

## One common model does not mean one universal layer

The shared backbone is common. The readout recipe is not.

Across the completed model-by-task matrix, the best hidden-state location changed with both task and
model. Masked-mean pooling dominated sentiment and toxicity. Spam split evenly between masked mean
and masked max. Toxicity and jailbreak detection selected different readouts even when they saw the
same input rows from the same model.

Each head therefore carries its own manifest:

- task and label contract;
- exact model and tokenizer revision;
- deterministic input renderer;
- token region and pooling method;
- hidden output or layer window;
- normalization and head family;
- calibration and operating thresholds;
- artifact hashes and benchmark evidence.

The runtime shares the forward pass, not an unsupported assumption that every task lives in the
same layer.

## Why this can be more practical than LLM-as-a-judge

For one isolated decision, a direct judge may be good enough or even more accurate. The architecture
becomes more useful when one workflow needs many decisions on the same input.

Instead of writing, tuning and running a separate prompt for every check, one local backbone pass can
serve a bank of narrow heads. Each head returns a continuous score, keeps its own threshold, and can
be installed, updated, disabled or rolled back without changing the backbone or the other heads.

With the model revision, renderer, readout, weights and threshold frozen, the score is also easier to
reproduce and audit than a generated answer. A temperature-zero judge can be deterministic too, but
it still depends on prompt wording, chat templates, label tokens and output handling.

This is a cost and control argument, not a claim that every probe beats every judge. When quality is
close, the shared head can be the better production choice because its marginal cost is tiny. When a
judge is materially better, as in the current authorization transfer result, the head needs better
data or should not be promoted.

## Where it can run

Today, IntentProbe can run as a local sidecar beside a hosted agent such as Claude or Codex. The
sidecar model exposes its own hidden states and produces local policy signals without access to the
hosted model's internals.

If the primary agent model is local and exposes hidden states, the same architecture can attach its
heads directly to that model. The model family is replaceable. Each replacement still needs its own
readout search, training and held-out validation because probe weights do not transfer automatically
between backbones.

## What is established, and what comes next

The current evidence establishes that one frozen 0.5B model can serve 16 real, independently trained
heads with nearly flat marginal latency and memory in this CPU environment. It also establishes that
quality is task-specific: toxicity favoured the trained readout, sentiment was competitive either
way, and the tested authorization transfer favoured the judge.

Two important comparisons remain open:

- one prompted call that asks the same model for all labels at once;
- a measured fleet of compact task-specific classifiers on the same machine and inputs.

Those tests will locate the full accuracy, latency and memory frontier. They do not change the result
already measured: once compatible activation heads exist, many of them can share one model forward
at very low marginal cost.

## Evidence and implementation record

- [Research manuscript: *One Forward, Sixteen Decisions*](../research/multitask-activation-benchmark/paper/DRAFT.md)
- [Complete implementable architecture disclosure](../research/multitask-activation-benchmark/ARCHITECTURE_DISCLOSURE_V0_1.md)
- [One-forward scaling result](../research/multitask-activation-benchmark/experiments/ONE-FORWARD-MULTIHEAD-EXP001/RESULT_SUMMARY.md)
- [Probe-versus-judge sentinel](../research/multitask-activation-benchmark/experiments/SENTINEL-JUDGE-VS-PROBE-EXP001/RESULT_SUMMARY.md)
- [Readout manifest schema](../research/schemas/intentprobe_readout_manifest.schema.json)
- [Multi-head receipt schema](../research/schemas/intentprobe_multihead_receipt.schema.json)

## Citation

Nicole BeWater. 2026. *IntentProbe: One Forward, Many Decisions. Public architecture note v0.1.0.*
MCPware, a product and research brand of GEASS LABS INC.

This release does not claim to invent activation probing, linear probing or shared-backbone
multi-head classification. It documents the IntentProbe architecture, its task-specific readout
contracts and the first measured 16-head local runtime built around them.
