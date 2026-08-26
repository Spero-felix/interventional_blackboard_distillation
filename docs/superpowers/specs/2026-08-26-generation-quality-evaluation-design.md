# Generation Quality Evaluation Design

## Goal

Add a reproducible generation-quality comparison on the same `dev` and
`diagnostic_holdout` histories used by the existing pipeline. The first
comparison covers the trained Student checkpoint, its corresponding unmodified
base model, and the frozen Teacher response already stored in `TeacherTrace`.

This evaluation complements the existing Stage C causal-controllability
evaluation. It does not change, replace, or merge with the Stage C metrics.

## Fixed protocol

- Each evaluated model produces exactly one normal response per history.
- Student and Base use identical prompt encoding, truncation, and deterministic
  decoding parameters.
- Teacher responses are read from `TeacherTrace.final_response`; the six-call
  Teacher pipeline is not rerun.
- Results for `dev` and `diagnostic_holdout` are reported separately. A pooled
  summary may be emitted only as a secondary appendix statistic.
- The LLM Judge model, prompt version, temperature, seed, and output contract
  are fixed within an evaluation run.
- Evaluated models are configured rather than hard-coded. The first version
  implements `teacher_trace`, `base_qwen`, and `student_checkpoint` response
  sources and leaves the registry open to additional local checkpoints.

## Architecture

Use an independent, artifact-driven workflow:

```text
fixed dev/holdout histories
    -> quality-generate
    -> responses.jsonl
    -> quality-judge
    -> judgments.jsonl + quality_report.json
    -> quality-human-export (optional)
    -> blinded_pairs.csv/jsonl + private_mapping.jsonl
    -> quality-human-summarize (optional)
    -> human_preference_report.json
```

Generation, LLM judging, human export, and human aggregation remain separate so
each expensive stage can be resumed or rerun without repeating earlier stages.
The implementation may reuse local storage, backend, Qwen-loading, prompting,
and progress-reporting utilities, but it must not create a runtime dependency on
the sibling `supervisor` repository.

Each stage also writes a small JSON manifest beside its row-oriented output.
The manifest contains the readable protocol version, selected splits, ordered
model registry, generation or Judge settings, seed, expected sample keys, and
artifact paths. Resume checks this manifest before accepting existing rows.

## Commands

### `quality-generate`

The command selects `dev` and `diagnostic_holdout` traces, generates one Base
and one Student response for each history, and copies the Teacher final response
into a common response artifact. It loads Base and Student sequentially and
releases one model before loading the next so both models do not need to reside
on the GPU simultaneously.

Successful rows are appended immediately. By default, an existing output is
rejected. With `--resume`, the command validates existing rows and skips keys
identified by `(split, example_id, model_id)`.

### `quality-judge`

The command scores each response independently. The Judge receives only the
fixed history and one anonymous candidate response; it never receives the
candidate's model ID, checkpoint path, response source, or experimental stage.
The command writes auditable per-response judgments before producing the
aggregate report.

### `quality-human-export`

The command optionally creates blinded pairwise comparisons for:

1. Student versus Base.
2. Student versus Teacher.
3. Base versus Teacher.

A fixed seed determines A/B placement. Within each `(split, model_pair)` group,
the exporter sorts the example IDs, applies a seeded shuffle, then alternates
orientation. This gives exact balance for even group sizes and a difference of
one for odd group sizes. The public review artifact and private identity mapping
are separate.

### `quality-human-summarize`

This optional command validates returned annotations and aggregates them. The
LLM Judge report is complete even when this command is never run.

## Response artifact

Each `responses.jsonl` row contains at least:

```json
{
  "protocol_version": "generation-quality-v1",
  "example_id": "example-1",
  "split": "dev",
  "model_id": "student",
  "response_source": "student_checkpoint",
  "history": {"turns": []},
  "response": "...",
  "generation_seed": 42,
  "generation_config": {
    "do_sample": false,
    "max_new_tokens": 256
  }
}
```

Teacher rows use `response_source: teacher_trace` and explicitly record that the
response was reused. They do not claim a new generation seed. Artifacts store
readable protocol versions and configurations, not generated content hashes, in
accordance with the repository's no-hash constraint.

Before comparison or aggregation, the evaluator requires the same history for
all model rows sharing `(split, example_id)`. Duplicate model rows, conflicting
histories, unknown splits, and unexpected missing model rows are explicit
validation errors or explicit coverage failures; none are silently discarded.

## Judge rubric

Adapt the evidence-grounded 1--5 rubric and structured-output pattern from
`/homeb/wangnianxiang/supervisor`, while implementing a local prompt and schema.
Remove the sibling project's safety dimension, safety cap, safety failure gate,
and reward formula completely.

The five dimensions are:

- `empathy`: accurate and context-specific recognition or validation of the
  seeker's emotional experience.
- `relevance`: grounding in the latest message and important visible history.
- `coherence`: clarity, consistency, logical flow, and conversational
  continuity.
- `effectiveness`: whether the response chooses the best immediate supportive
  move for the dialogue stage and the seeker's readiness. Listening or staying
  with emotion can be more effective than giving advice.
- `non_coerciveness`: respect for autonomy, boundaries, pace, and choice.

The Judge returns one integer score from 1 through 5 and one concise,
text-grounded reason for every dimension, plus a short summary reason. It does
not return `overall`. Code computes:

```text
overall = mean(empathy, relevance, coherence, effectiveness, non_coerciveness)
```

No safety score, threshold, cap, penalty, or derived safety field appears in the
prompt, schema, artifacts, or report.

## LLM Judge report

Primary summaries are separate for `dev` and `diagnostic_holdout`. For each
model and split, report:

- expected, successful, failed, and missing counts plus coverage rate;
- mean score for every dimension;
- mean `overall` score;
- auditable per-example scores and reasons.

For Student--Base and Student--Teacher, report same-history paired score
differences for every dimension and `overall`, together with win, tie, and loss
proportions. The report may also include Base--Teacher for completeness. A tie
means equal numeric scores on the compared field.

The report records the Judge model identity and whether it matches the model
used to create the Teacher traces when that information can be resolved. A
match emits a warning and an audit flag but does not block execution.

## Human review artifacts

The public artifact contains:

```text
pair_id, split, example_id, history, response_a, response_b,
preference, reviewer_id, comment
```

`preference` accepts only `A`, `B`, or `tie`; `reviewer_id` and `comment` are
optional. The public artifact contains no model names, response-source labels,
checkpoint paths, or generation metadata.

The exporter assigns sequential `pair_id` values after deterministic group and
example ordering; it does not derive IDs from content hashes. The private
mapping records `pair_id`, `model_a`, and `model_b`. Multiple reviewers may
submit separate rows for the same `pair_id`. Aggregation preserves all valid
votes rather than forcing disagreements into one synthetic label. The human
report contains raw vote counts, win and tie rates, valid annotation counts, and
missing or invalid counts for each split and model pair.

## Failure and resume behavior

- Every successful generation or judgment is durably appended before the next
  item starts.
- Per-item failures are written to an explicit failure ledger with the sample
  key, exception type, and message.
- Resume validates the existing protocol version, model registry, split,
  generation or Judge settings, and row schemas before skipping completed keys.
- Aggregation reports incomplete coverage and never substitutes successful rows
  for the intended full evaluation set.
- Existing artifacts are not overwritten without an explicit resume or output
  choice.

## Configuration boundary

Use a dedicated generation-quality evaluation configuration rather than adding
Judge settings to the Teacher protocol or Qwen training configuration. The
configuration names the evaluated models and their response-source types,
selects the fixed Judge role, and freezes generation and randomization settings.
The first version validates the canonical Teacher/Base/Student set while keeping
the model registry extensible to more local checkpoints.

## Verification

Offline tests cover:

- strict response and five-dimension judgment schemas;
- arithmetic computation of `overall`;
- split isolation and same-history paired comparisons;
- Teacher response reuse without a model call;
- identical Base and Student prompt/decoding settings;
- absence of model identity and safety language from Judge inputs;
- structured-output retry, failure ledgers, non-overwrite behavior, and resume;
- incomplete and inconsistent model coverage detection;
- deterministic anonymous A/B assignment and within-group placement balance;
- public artifact anonymity and private mapping correctness;
- human preference validation and aggregation, including multiple reviewers;
- a small scripted end-to-end workflow requiring neither a GPU nor a live API.

Real-model generation and live Judge calls remain explicit runtime smoke tests,
not requirements for the offline test suite.

## Non-goals

- Do not alter the Stage C causal evaluation or its report.
- Do not rerun the Teacher pipeline.
- Do not add safety evaluation or safety-derived fields.
- Do not make human annotation mandatory for the LLM Judge report.
- Do not add OpenAI-compatible evaluated-model adapters in the first version.
- Do not copy the `supervisor` architecture or depend on its Python package.
