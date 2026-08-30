# Teacher Candidate Debiasing and Parsing Design

## Goal

Remove the fixed S1/S2/S3-to-role/seed coupling, prevent selector presentation-order bias while preserving reproducibility, and stop Markdown-fenced candidate JSON from entering training data as natural-language responses. Add an A=1 experiment configuration without rewriting historical runs.

## Confirmed behavior

- All three candidate generations use one semantic role: `candidate`.
- Candidate generation does not use or prompt for position-bound seeds.
- Selector presentation order is a deterministic permutation derived from the protocol version and `example_id`.
- Candidate identity remains stable: planner strategy S1 still produces candidate ID `1`, S2 produces `2`, and S3 produces `3`.
- `TeacherTrace.candidates` remains in canonical candidate-ID order. Only the list presented to `final_selector` is permuted.
- Valid fenced JSON is parsed as structured candidate output. Malformed structured-looking output triggers the existing schema retry instead of becoming a response string.
- Historical candidate artifacts with integer `seed` values remain readable; newly generated candidates store `seed: null`.
- A new experiment config uses one Stage A epoch and leaves existing configs and run directories unchanged.

## Architecture

### Candidate role and cache identity

`candidate_1`, `candidate_2`, and `candidate_3` are replaced by one `candidate` prompt/config role. `StructuredCaller.call` gains an optional `cache_variant` argument, and the cache digest includes `(example_id, role, cache_variant, protocol namespace)`. Teacher candidate calls pass their stable strategy IDs (`S1`, `S2`, or `S3`) as the variant. This keeps three cache entries per example without assigning different semantic roles to candidate positions.

The cached payload records the variant for diagnostics. Non-candidate callers omit it and preserve their current cache behavior.

### Seed compatibility

`candidate_seeds` is removed from current Teacher YAML configurations and from active generation logic. Candidate prompts no longer contain or freeze a seed field. `Candidate.seed` becomes optional with a `None` default so existing JSONL traces remain schema-compatible while new traces explicitly represent the absence of a provider seed.

General backend support for seeded providers remains available through `ModelConfig.supports_seed`; this design removes only the fixed candidate-position mapping.

### Deterministic selector presentation

Before calling `final_selector`, the Teacher creates a separate presentation list sorted by a SHA-256 rank derived from:

1. protocol version,
2. example identity, and
3. candidate identity.

This gives the same example the same presentation order across reruns and distributes candidate IDs across selector positions over a dataset. If `example_id` is unavailable, the fallback identity is derived from the canonical candidate payloads so the operation remains deterministic.

The selector still returns `selected_candidate_id`; lookup is performed against the canonical candidates, so order changes cannot alter identity semantics.

### Plain-text and fenced-JSON parsing

The non-provider-JSON candidate adapter classifies output into three paths:

- A JSON object, either bare or inside a Markdown fence, is returned as structured JSON for normal Candidate validation.
- A JSON string or ordinary natural-language text is wrapped into the frozen candidate metadata with default goal/act metadata.
- Output that looks structured but is malformed, including malformed fences, is passed through unchanged so `StructuredCaller` records the validation failure and performs its one configured retry.

A valid JSON fence may have surrounding whitespace or a short provider preamble. Multiple or ambiguous fenced blocks are not guessed; they take the retry path.

The Teacher protocol version is bumped so existing contaminated cache entries cannot be reused under the new configuration.

## Training configuration

Create `configs/experiments/ab2000-a1-b5-lr-8e-5.yaml` with Stage A set to one epoch, Stage B set to five epochs, and Stage C disabled. Set `scheduler_total_steps: 1128`, matching six planned epochs at 188 optimizer steps per epoch for the current 1,500-row training split.

This is a new experiment rather than a universal training default. Stage B continues to include response cross-entropy plus slot alignment, so A=1 reduces repeated response-only reinforcement but does not remove response learning. Existing A=5 results remain intact for comparison.

Automatically restoring the best checkpoint between arbitrary stages is intentionally out of scope. With A fixed to one epoch, the A-to-B handoff is unambiguous. A later B-to-C experiment should separately address best-checkpoint handoff before being treated as a production protocol.

## Error handling and compatibility

- Cache variants are optional and do not affect existing non-candidate cache keys.
- Missing or unknown candidate IDs from the selector remain fatal.
- Malformed fenced JSON consumes the existing schema retry budget and remains fatal if the retry is also invalid.
- Existing traces with numeric candidate seeds validate unchanged.
- Existing current-protocol caches are not mutated or deleted; the protocol namespace change makes them unreachable to the new run.
- Existing experiment files and run outputs are not edited.

## Tests

1. A fenced Candidate JSON object, including an observed preamble-plus-fence shape, produces the inner natural response rather than the fence text.
2. Malformed fenced JSON triggers a schema retry and is never accepted as `Candidate.response`.
3. Plain natural-language candidate output is still wrapped successfully.
4. Three candidate calls use the same `candidate` role, pass no provider seed, and retain distinct IDs/strategies/responses.
5. Candidate calls for one example occupy distinct cache variants and replay correctly from cache.
6. Selector presentation is stable for the same example and covers all candidate IDs across all three presentation positions over a deterministic sample of example IDs.
7. Canonical trace candidate order and selected-candidate lookup remain unchanged.
8. Historical Candidate payloads with integer seeds and new payloads with `null` seeds both validate.
9. Teacher, prompting, config, intervention, cache, and full unit suites pass.

## Non-goals

- Rebalancing planner strategy frequencies.
- Changing the selector quality rubric.
- Repairing historical artifact files in place.
- Making DeepSeek generation itself deterministic when the provider does not support seeds.
- Automatically selecting or restoring the best checkpoint between every training stage.
