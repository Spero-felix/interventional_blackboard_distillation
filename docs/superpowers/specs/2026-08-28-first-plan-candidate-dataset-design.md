# First-Plan Candidate Dataset Design

## Goal

Build a new 2,000-example Teacher trace dataset from the existing completed
traces by manually fixing every natural path to the first planned candidate
(`S1`). Preserve the existing train, development, and diagnostic holdout split,
then use the repository's existing intervention, anchor, training, and
evaluation pipeline without changing the main `ibd` CLI.

The source dataset is
`artifacts/full-2000/teacher-traces.jsonl`. It currently contains 1,500 train,
200 dev, and 300 diagnostic holdout examples. The conversion must not overwrite
this file or any existing artifact.

## Interface

Add one standalone script:

```text
scripts/select_first_plan_candidate.py
```

Its command-line interface is:

```bash
$PY scripts/select_first_plan_candidate.py \
  --input artifacts/full-2000/teacher-traces.jsonl \
  --output artifacts/full-2000-first-plan/teacher-traces.jsonl
```

The candidate choice is deliberately fixed to `S1`; the script does not expose
a general strategy selector. This keeps its behavior identical to the intended
manual construction rule: always take the first item in the Teacher plan.

The script refuses to overwrite an existing output file. Parent directories may
be created automatically.

## Conversion Rules

For every input row:

1. Validate it as a `TeacherTrace` before transformation.
2. Find the unique candidate whose `strategy_id` is `S1`. Do not rely on list
   position alone.
3. Confirm that this candidate's strategy equals the first strategy in
   `trace.plan.strategies`.
4. Rebuild `final_selection` canonically from that candidate, using the
   candidate's own response goal, response act, strategy, ID, and response.
5. Set `final_response` to the same candidate response.
6. Preserve `example_id`, `split`, `history`, `state_analysis`, `state`, `plan`,
   all three `candidates`, and `call_records` unchanged.
7. Validate the complete transformed row as a `TeacherTrace` again.

The output ordering is identical to the input ordering. No examples are sampled,
duplicated, moved between splits, or discarded.

## Dataset-Level Validation and Reporting

Before writing output, the script verifies:

- the input contains exactly 2,000 traces;
- all `example_id` values are unique;
- every trace has exactly one `S1` candidate;
- the split counts are exactly 1,500 train, 200 dev, and 300 diagnostic holdout;
- every transformed trace selects `S1` and passes the full schema provenance
  checks.

Any failed condition terminates the command without creating a partial output.
On success, the script writes the complete JSONL atomically and prints:

- total row count;
- split counts;
- selected strategy-ID distribution before conversion;
- selected strategy-ID distribution after conversion;
- output path.

## Downstream Data Flow

The new trace file is the sole source for the new experiment. Existing artifacts
must not be reused where their semantics depend on the old final selection.

```text
existing 2,000 Teacher traces
  -> standalone S1 conversion script
  -> new canonical Teacher traces
  -> existing build-interventions command
  -> new intervention JSONL and manifest
  -> existing precompute-anchors command
  -> new anchor artifact
  -> existing train-pipeline command
  -> existing dev and diagnostic-holdout evaluation
```

Interventions must be rebuilt. Old intervention records can contain the old
`full_response`, old original PLAN condition, and STATE-counterfactual outputs
derived from a different natural selection. Anchors must also be rebuilt because
the natural PLAN payload changes to the S1 candidate's goal and act.

The existing pipeline continues to define split usage: train examples fit the
model, dev examples support checkpoint selection and causal evaluation, and the
300 diagnostic holdout examples remain excluded from fitting.

## Tests

Tests for the standalone script cover:

- an S2-selected input trace is converted to its S1 candidate;
- an already-S1 trace remains semantically unchanged;
- all non-selection trace fields and input order are preserved;
- lookup uses `strategy_id == "S1"` even if candidate list order changes;
- missing or duplicate S1 candidates fail before output is written;
- duplicate example IDs fail;
- unexpected total or split counts fail;
- an existing output path is not overwritten;
- a valid 2,000-row fixture produces the required split and selection summary.

The implementation follows test-first development. Unit tests exercise pure
conversion and dataset-validation functions; one command-level test checks file
creation and overwrite protection without invoking any model or GPU.

## Operational Documentation

Add a focused README section with copy-paste commands for:

1. converting and validating the traces on CPU;
2. rebuilding interventions with the existing Teacher configuration;
3. precomputing train and dev anchors;
4. launching the A/B/C training pipeline in a new run directory/name;
5. running the existing evaluation workflow against the new artifacts.

The commands use a dedicated `artifacts/full-2000-first-plan/` namespace and a
new run name so the current 2,000-example experiment remains intact.
