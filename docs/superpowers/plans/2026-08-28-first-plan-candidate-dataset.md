# First-Plan Candidate Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone, CPU-only script that converts the existing 2,000 Teacher traces to canonical S1-selected traces and document the exact commands for rebuilding interventions, anchors, A/B/C training, and evaluation.

**Architecture:** Keep the existing `ibd` CLI and pipeline unchanged. Put pure trace conversion, dataset validation, summary construction, atomic JSONL persistence, and a small argparse entry point in one focused standalone script; test it by loading that script as a module and by invoking its `main` function with temporary files.

**Tech Stack:** Python 3.11, Pydantic v2 `TeacherTrace`/`FinalSelection` schemas, standard-library `argparse`, `json`, `collections`, `os`, `pathlib`, `tempfile`, and pytest.

## Global Constraints

- Do not modify `src/ibd/cli.py` or add an `ibd` subcommand.
- The selection rule is fixed to the unique candidate with `strategy_id == "S1"`; do not expose a configurable selector.
- Preserve input order and every field except `final_selection` and `final_response`.
- Require exactly 2,000 unique examples split as train 1,500, dev 200, and diagnostic holdout 300.
- Refuse to overwrite the source, an existing output, or any current artifact.
- Validate every row before and after conversion and finish all dataset validation before publishing output.
- Rebuild interventions and anchors; never reuse artifacts derived from the old final selections.
- Preserve all unrelated pre-existing worktree changes.

---

## File Structure

- Create `scripts/select_first_plan_candidate.py`: complete standalone conversion command and importable pure functions.
- Create `tests/test_select_first_plan_candidate.py`: unit and command-level coverage using small traces plus a generated 2,000-row dataset.
- Modify `README.md`: append the copy-paste first-plan experiment workflow without changing unrelated documentation.

### Task 1: Pure S1 Conversion and Dataset Validation

**Files:**
- Create: `scripts/select_first_plan_candidate.py`
- Create: `tests/test_select_first_plan_candidate.py`

**Interfaces:**
- Consumes: `ibd.schemas.TeacherTrace` and `ibd.schemas.FinalSelection`.
- Produces: `select_s1(trace: TeacherTrace) -> TeacherTrace`, `convert_dataset(traces: Sequence[TeacherTrace]) -> tuple[list[TeacherTrace], ConversionSummary]`, and frozen dataclass `ConversionSummary(total: int, splits: dict[str, int], selected_before: dict[str, int], selected_after: dict[str, int])`.

- [ ] **Step 1: Create the test module loader and conversion fixtures**

Load the standalone script without turning `scripts/` into an application package, and build valid traces through the existing offline Teacher fixture:

```python
import importlib.util
import sys
from pathlib import Path

from ibd.teacher import TeacherRunner

SCRIPT = Path(__file__).parents[1] / "scripts" / "select_first_plan_candidate.py"
SPEC = importlib.util.spec_from_file_location("select_first_plan_candidate", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

def make_trace(history, app_config, backend, example_id="e-1", split="train"):
    return TeacherRunner(backend, app_config).run(example_id, history, split=split)
```

- [ ] **Step 2: Write failing tests for canonical S1 selection**

Add tests asserting that the fixture's originally selected S2 response becomes the S1 candidate response, that `FinalSelection` uses the S1 candidate's own goal and act, and that all fields other than `final_selection` and `final_response` have identical JSON values. Add a second test that reverses the candidate array before conversion and still selects by ID:

```python
def test_select_s1_rebuilds_only_final_fields(history, app_config):
    trace = make_trace(history, app_config, ScriptedBackend())
    converted = module.select_s1(trace)
    s1 = next(item for item in trace.candidates if item.strategy_id == "S1")
    assert converted.final_selection.selected_candidate_id == s1.candidate_id
    assert converted.final_selection.response_goal == s1.response_goal
    assert converted.final_selection.response_act == s1.response_act
    assert converted.final_response == s1.response
    before = trace.model_dump(mode="json")
    after = converted.model_dump(mode="json")
    before.pop("final_selection"); before.pop("final_response")
    after.pop("final_selection"); after.pop("final_response")
    assert after == before
```

Add an already-S1 case by canonicalizing the fixture once and passing that result
through `select_s1` again; assert the second result equals the first model dump.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_select_first_plan_candidate.py -q
```

Expected: collection fails because `scripts/select_first_plan_candidate.py` does not exist.

- [ ] **Step 4: Implement minimal canonical conversion**

Create `select_s1` with these operations:

```python
def select_s1(trace: TeacherTrace) -> TeacherTrace:
    matches = [item for item in trace.candidates if item.strategy_id == "S1"]
    if len(matches) != 1:
        raise ValueError(
            f"example {trace.example_id} must contain exactly one S1 candidate; "
            f"found {len(matches)}"
        )
    candidate = matches[0]
    if trace.plan.strategies[0] != candidate.strategy:
        raise ValueError(f"example {trace.example_id} S1 candidate does not match plan[0]")
    payload = trace.model_dump(mode="json")
    payload["final_selection"] = FinalSelection.from_candidate(candidate).model_dump(mode="json")
    payload["final_response"] = candidate.response
    return TeacherTrace.model_validate(payload)
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Task 1 focused pytest command. Expected: the selection and preservation tests pass.

- [ ] **Step 6: Write failing dataset-validation tests**

Add tests for duplicate IDs, wrong total, wrong split counts, and final summary. Avoid 2,000 Teacher model calls: create one valid trace per split, then clone it with `model_copy(update={"example_id": ...})`. Expected summary:

```python
assert summary.total == 2000
assert summary.splits == {"train": 1500, "dev": 200, "diagnostic_holdout": 300}
assert summary.selected_after == {"S1": 2000}
```

For malformed candidate collections, construct raw payloads and assert `select_s1` rejects zero or two S1 matches before publishing a result; use `TeacherTrace.model_construct` only at this unit boundary because the production file reader still performs full schema validation.

- [ ] **Step 7: Run new validation tests and verify RED**

Run the focused pytest command. Expected: failures report missing `convert_dataset` or `ConversionSummary`.

- [ ] **Step 8: Implement dataset validation and deterministic summaries**

Use constants and sorted dictionaries so diagnostics are stable:

```python
EXPECTED_TOTAL = 2000
EXPECTED_SPLITS = {"train": 1500, "dev": 200, "diagnostic_holdout": 300}

@dataclass(frozen=True)
class ConversionSummary:
    total: int
    splits: dict[str, int]
    selected_before: dict[str, int]
    selected_after: dict[str, int]
```

`convert_dataset` must reject the first duplicate ID, validate total and exact split equality, call `select_s1` in input order, and assert every output selects S1 before returning the list and summary.

- [ ] **Step 9: Run focused tests and verify GREEN**

Run the Task 1 focused pytest command. Expected: all pure-function tests pass.

- [ ] **Step 10: Commit Task 1**

```bash
git add scripts/select_first_plan_candidate.py tests/test_select_first_plan_candidate.py
git commit -m "feat: canonicalize teacher traces to first plan candidate"
```

### Task 2: Safe Standalone File Command

**Files:**
- Modify: `scripts/select_first_plan_candidate.py`
- Modify: `tests/test_select_first_plan_candidate.py`

**Interfaces:**
- Consumes: Task 1 `convert_dataset` and `ConversionSummary`.
- Produces: `read_traces(path: Path) -> list[TeacherTrace]`, `write_jsonl_atomically(path: Path, traces: Sequence[TeacherTrace]) -> None`, `build_parser() -> argparse.ArgumentParser`, and `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing command tests**

Create a valid 2,000-row JSONL under `tmp_path`, call `module.main(["--input", str(source), "--output", str(target)])`, and assert:

```python
assert result == 0
rows = [json.loads(line) for line in target.read_text().splitlines()]
assert len(rows) == 2000
assert {row["final_selection"]["selected_strategy_id"] for row in rows} == {"S1"}
```

Capture stdout and assert it contains total, exact split counts, before/after selection distributions, and the target path. Add tests that an existing target and identical resolved input/output paths raise `FileExistsError` or `ValueError`, and that invalid input leaves no target or temporary sibling.

- [ ] **Step 2: Run command tests and verify RED**

Run the focused pytest command. Expected: failures report missing `main` and file helpers.

- [ ] **Step 3: Implement validated reading and atomic no-overwrite writing**

`read_traces` must parse nonblank JSONL lines and include line numbers in JSON or schema errors. `write_jsonl_atomically` must create a temporary file in the target directory, flush and `os.fsync`, then publish using `os.link(temp, target)` so an existing target fails atomically; always unlink the temporary file in `finally`. Create the parent only after conversion succeeds.

Use `json.dumps(trace.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)` for each line. Do not use the existing truncating `storage.write_jsonl`, because this command requires no-overwrite publication.

- [ ] **Step 4: Implement argparse entry point and stable JSON summary**

The parser exposes only required `--input` and `--output`. Before reading, resolve both paths without requiring existence and reject equality. Before all conversion work, reject an already existing output. Print one JSON object using:

```python
print(json.dumps({
    "total": summary.total,
    "splits": summary.splits,
    "selected_before": summary.selected_before,
    "selected_after": summary.selected_after,
    "output": str(output_path),
}, ensure_ascii=False, sort_keys=True))
```

End the file with `raise SystemExit(main())`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Task 1/2 focused pytest command. Expected: all script tests pass and no temporary files remain.

- [ ] **Step 6: Run the real source as a read-only dry validation through pure functions**

Do not create the requested artifact for the user. Import the script and run `read_traces` plus `convert_dataset` against `artifacts/full-2000/teacher-traces.jsonl`; print the returned summary only. Expected: total 2,000, splits 1,500/200/300, before distribution S1=236/S2=1,363/S3=401, after distribution S1=2,000.

- [ ] **Step 7: Commit Task 2**

```bash
git add scripts/select_first_plan_candidate.py tests/test_select_first_plan_candidate.py
git commit -m "feat: add safe first-plan dataset command"
```

### Task 3: Operational Workflow Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_select_first_plan_candidate.py`

**Interfaces:**
- Consumes: standalone script and existing `ibd.cli` commands.
- Produces: copy-paste commands that create only `artifacts/full-2000-first-plan/` and a uniquely named run.

- [ ] **Step 1: Add the CPU conversion command to README**

Append a section named `固定第一个 PLAN 候选的 2000 条实验` using:

```bash
export PYTHONPATH=src
PY=/home/wangnianxiang/supervisor/.venv/bin/python

$PY scripts/select_first_plan_candidate.py \
  --input artifacts/full-2000/teacher-traces.jsonl \
  --output artifacts/full-2000-first-plan/teacher-traces.jsonl
```

State that the expected summary is 2,000 rows, 1,500/200/300 splits, and S1=2,000.

- [ ] **Step 2: Document intervention and anchor rebuilding**

Add the exact existing CLI commands:

```bash
$PY -m ibd.cli build-interventions \
  --config configs/deepseek_teacher.yaml \
  --input artifacts/full-2000-first-plan/teacher-traces.jsonl \
  --output artifacts/full-2000-first-plan/interventions.jsonl \
  --manifest artifacts/full-2000-first-plan/intervention-manifest.json \
  --global-seed 42

CUDA_VISIBLE_DEVICES=0 $PY -m ibd.cli precompute-anchors \
  --config configs/experiments/c2000-from-b-lr-8e-5.yaml \
  --traces artifacts/full-2000-first-plan/teacher-traces.jsonl \
  --interventions artifacts/full-2000-first-plan/interventions.jsonl \
  --output artifacts/full-2000-first-plan/anchors-train-dev.safetensors \
  --original-splits train dev \
  --diagnostic-state-field readiness \
  --global-seed 42
```

Explicitly warn that `build-interventions` performs Teacher API calls and the old interventions are incompatible.

- [ ] **Step 3: Document full A/B/C training and evaluation**

Use the config whose epochs enable A, B, and C:

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m ibd.cli train-pipeline \
  --run-name first-plan-2000-lr-8e-5 --seed 42 \
  --config configs/experiments/c2000-from-b-lr-8e-5.yaml \
  --traces artifacts/full-2000-first-plan/teacher-traces.jsonl \
  --interventions artifacts/full-2000-first-plan/interventions.jsonl \
  --anchors artifacts/full-2000-first-plan/anchors-train-dev.safetensors
```

Document an evaluation command template with the checkpoint selected after training and output report name explained in prose; explain how to locate it in `runs/first-plan-2000-lr-8e-5/`.

- [ ] **Step 4: Run focused and complete offline tests**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_select_first_plan_candidate.py -q
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest -q
```

Expected: all tests pass. If unrelated pre-existing worktree changes cause a failure, isolate and report it rather than modifying unrelated files.

- [ ] **Step 5: Check formatting, script help, and diff scope**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python scripts/select_first_plan_candidate.py --help
git diff --check -- scripts/select_first_plan_candidate.py tests/test_select_first_plan_candidate.py README.md
git status --short
```

Expected: help lists only `--input` and `--output`; diff check is clean; no generated first-plan artifact is present.

- [ ] **Step 6: Commit documentation and verified result**

```bash
git add README.md scripts/select_first_plan_candidate.py tests/test_select_first_plan_candidate.py
git commit -m "docs: add first-plan training workflow"
```

Because README already has unrelated user changes, inspect its staged diff and stage only this task's hunk if necessary; never commit unrelated lines.
