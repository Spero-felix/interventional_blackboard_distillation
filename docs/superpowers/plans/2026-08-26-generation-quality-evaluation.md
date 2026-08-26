# Generation Quality Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable Student/Base/Teacher generation-quality comparison on fixed dev and diagnostic holdout histories, with a five-dimension blinded LLM Judge and an optional human pairwise review loop.

**Architecture:** Build a separate artifact-driven workflow alongside the existing Stage C evaluation. A dedicated quality configuration registers response sources; generation writes normalized response rows; judging consumes only anonymous history/response pairs; pure report functions aggregate split-specific paired results; optional human export and summarize commands operate on the same response artifact.

**Tech Stack:** Python 3.11, Pydantic 2, PyYAML, PyTorch/Transformers/PEFT for local generation, the existing OpenAI-compatible backend, stdlib JSON/CSV/statistics, pytest.

## Global Constraints

- Work directly in the current checkout; do not create another worktree.
- Preserve the user's existing changes in `configs/deepseek_teacher.yaml`, `src/ibd/cli.py`, and `tests/test_cli.py`; merge CLI edits carefully and never revert unrelated lines.
- Generate exactly one response per `(split, example_id, model_id)`.
- Reuse `TeacherTrace.final_response`; never rerun the Teacher pipeline.
- Keep `dev` and `diagnostic_holdout` primary reports separate.
- Use only `empathy`, `relevance`, `coherence`, `effectiveness`, and `non_coerciveness`; add no safety score, safety gate, cap, penalty, or safety-derived field.
- Compute `overall` in code as the arithmetic mean of the five scores.
- Base and Student share histories, chat-template semantics, history truncation policy, and deterministic decoding parameters. Only Student receives trained IBD structure tokens.
- Do not create a runtime dependency on `/homeb/wangnianxiang/supervisor`.
- Do not create or persist content/config hashes outside the existing backend cache implementation.
- Keep tests limited to protocol-critical behavior and one offline CLI smoke path.

---

## File Structure

- Create `src/ibd/quality.py`: strict quality configuration and artifact schemas, manifest validation, pure score aggregation.
- Create `src/ibd/quality_generation.py`: Teacher/Base/Student adapters, sequential model lifecycle, durable response generation.
- Create `src/ibd/quality_judge.py`: local five-dimension prompt, structured Judge call, durable judgment generation.
- Create `src/ibd/quality_human.py`: deterministic blinded CSV export, private mapping, optional annotation aggregation.
- Create `src/ibd/quality_cli.py`: four command parsers and command orchestration without enlarging `cli.py` further.
- Modify `src/ibd/qwen.py`: public Base generation loader and public Student checkpoint restore helper.
- Modify `src/ibd/student_data.py`: Base chat-generation encoding without IBD structure tokens.
- Modify `src/ibd/cli.py`: register and dispatch the four quality commands while preserving resumable Teacher work.
- Create `configs/quality_eval.yaml`: canonical Teacher/Base/Student and fixed Judge example.
- Create `tests/test_quality_evaluation.py`: focused protocol tests and offline smoke flow.
- Modify `README.md`: document the workflow, artifacts, and representative commands.

---

### Task 1: Quality configuration, schemas, and pure aggregation

**Files:**
- Create: `src/ibd/quality.py`
- Create: `tests/test_quality_evaluation.py`

**Interfaces:**
- Produces: `QualityEvalConfig.from_yaml(path) -> QualityEvalConfig`.
- Produces: `QualityResponse`, `QualityJudgment`, `QualityManifest`, `QualityFailure` Pydantic models.
- Produces: `compute_overall(scores: QualityScores) -> float`.
- Produces: `build_quality_report(judgments, expected_keys, model_ids) -> dict[str, object]`.
- Consumes: existing `BackendConfig`, `ModelConfig`, `CallRecord`, and `History`.

- [ ] **Step 1: Write one focused schema/aggregation test**

```python
def test_quality_report_keeps_splits_separate_and_computes_paired_results(history):
    from ibd.quality import (
        QualityJudgment,
        QualityScores,
        build_quality_report,
        compute_overall,
    )

    def row(split, model_id, empathy):
        scores = QualityScores(
            empathy=empathy,
            relevance=4,
            coherence=4,
            effectiveness=4,
            non_coerciveness=4,
        )
        return QualityJudgment(
            protocol_version="generation-quality-v1",
            example_id="e-1",
            split=split,
            model_id=model_id,
            scores=scores,
            overall=compute_overall(scores),
            dimension_reasons={name: "evidence" for name in scores.model_fields},
            short_reason="summary",
            call_records=[],
        )

    judgments = [
        row("dev", "student", 5),
        row("dev", "base", 3),
        row("diagnostic_holdout", "student", 2),
        row("diagnostic_holdout", "base", 4),
    ]
    expected = {
        (item.split, item.example_id, item.model_id) for item in judgments
    }
    report = build_quality_report(judgments, expected, ["student", "base"])
    assert report["splits"]["dev"]["paired"]["student__vs__base"]["empathy"]["mean_delta"] == 2.0
    assert report["splits"]["diagnostic_holdout"]["paired"]["student__vs__base"]["empathy"]["mean_delta"] == -2.0
    assert "safety" not in str(report).lower()
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_quality_evaluation.py::test_quality_report_keeps_splits_separate_and_computes_paired_results -q`

Expected: FAIL because `ibd.quality` does not exist.

- [ ] **Step 3: Implement strict schemas and aggregation**

Implement these central contracts in `quality.py`:

```python
QualitySplit = Literal["dev", "diagnostic_holdout"]
ResponseSource = Literal["teacher_trace", "base_qwen", "student_checkpoint"]
QUALITY_DIMENSIONS = (
    "empathy", "relevance", "coherence", "effectiveness", "non_coerciveness"
)

class GenerationSettings(StrictModel):
    max_new_tokens: int = Field(default=256, gt=0)
    do_sample: Literal[False] = False
    seed: int = 42

class EvaluatedModel(StrictModel):
    model_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    source: ResponseSource
    training_config: Path | None = None
    checkpoint: Path | None = None
    run_name: str | None = None

class QualityEvalConfig(StrictModel):
    protocol_version: Literal["generation-quality-v1"]
    splits: tuple[QualitySplit, ...] = ("dev", "diagnostic_holdout")
    models: list[EvaluatedModel] = Field(min_length=2)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    judge: ModelConfig
    judge_seed: int = 4242
    judge_prompt_version: Literal["quality-judge-v1"] = "quality-judge-v1"
    human_seed: int = 2026
    schema_retries: int = Field(default=1, ge=0, le=1)
```

Add a model validator that requires unique `model_id` values, requires exactly
one `teacher_trace` source in v1, requires `training_config` for Base, requires
`training_config`, `checkpoint`, and `run_name` for Student, and rejects those
fields for Teacher. `QualityResponse.generation_seed` is `int | None`, so copied
Teacher rows use `None`. Validate `QualityJudgment.overall` against
`compute_overall`. Resolve relative training-config and checkpoint paths against
the directory containing the quality YAML, not the process working directory.

`build_quality_report` must group by split, emit model coverage and dimension
means, intersect same-history keys for every model pair, and calculate
`mean_delta`, `wins`, `ties`, `losses`, and corresponding proportions for each
dimension plus `overall`.

- [ ] **Step 4: Run the focused test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_quality_evaluation.py::test_quality_report_keeps_splits_separate_and_computes_paired_results -q`

Expected: PASS.

- [ ] **Step 5: Commit the schema and aggregation slice**

```bash
git add src/ibd/quality.py tests/test_quality_evaluation.py
git commit -m "feat: add generation quality contracts"
```

---

### Task 2: Fair Base and Student generation adapters

**Files:**
- Modify: `src/ibd/student_data.py`
- Modify: `src/ibd/qwen.py`
- Create: `src/ibd/quality_generation.py`
- Modify: `tests/test_quality_evaluation.py`

**Interfaces:**
- Consumes: `QualityEvalConfig`, `QualityResponse`, `QualityManifest`, `QualityFailure` from Task 1.
- Produces: `encode_base_generation_prompt(tokenizer, history, max_length) -> dict[str, Tensor]`.
- Produces: `load_qwen_base_for_generation(config, device=0) -> BaseQwen`.
- Produces: `restore_qwen_for_inference(config, checkpoint, run_name, device=0) -> LoadedQwen`.
- Produces: `generate_quality_responses(config, traces, output_path, manifest_path, failure_path, resume, continue_on_error, device, adapter_factories=None) -> int`.

- [ ] **Step 1: Add the core generation-boundary test**

Use a fake tokenizer and fake adapters so the test needs no Torch model:

```python
def test_generation_reuses_teacher_and_keeps_ibd_tokens_out_of_base(
    tmp_path, history, app_config
):
    from ibd.quality import QualityEvalConfig
    from ibd.quality_generation import generate_quality_responses
    from ibd.storage import read_jsonl
    from ibd.teacher import TeacherRunner

    trace = TeacherRunner(ScriptedBackend(), app_config).run(
        "e-1", history, split="dev"
    )
    calls = []

    class FakeAdapter:
        def __init__(self, model_id, uses_structure_tokens):
            self.model_id = model_id
            self.uses_structure_tokens = uses_structure_tokens
        def generate(self, seen_history, settings):
            calls.append((self.model_id, seen_history, self.uses_structure_tokens, settings))
            return f"{self.model_id} response"
        def close(self):
            return None

    config = quality_config_fixture(tmp_path)
    generate_quality_responses(
        config,
        [trace],
        output_path=tmp_path / "responses.jsonl",
        manifest_path=tmp_path / "responses.manifest.json",
        failure_path=tmp_path / "generation_failures.jsonl",
        resume=False,
        continue_on_error=False,
        device=0,
        adapter_factories={
            "base_qwen": lambda spec, device: FakeAdapter(spec.model_id, False),
            "student_checkpoint": lambda spec, device: FakeAdapter(spec.model_id, True),
        },
    )
    rows = read_jsonl(tmp_path / "responses.jsonl")
    assert {row["model_id"] for row in rows} == {"teacher", "base", "student"}
    assert next(row for row in rows if row["model_id"] == "teacher")["response"] == trace.final_response
    assert calls[0][2] is False and calls[1][2] is True
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_quality_evaluation.py::test_generation_reuses_teacher_and_keeps_ibd_tokens_out_of_base -q`

Expected: FAIL because generation adapters do not exist.

- [ ] **Step 3: Add fair model loading and prompt encoding**

In `student_data.py`, add `encode_base_generation_prompt` using
`history_messages` and `tokenizer.apply_chat_template(...,
add_generation_prompt=True)` without calling `add_ibd_tokens` or appending IBD
tokens. Enforce the same `max_length` history budget as Student.

In `qwen.py`, add:

```python
@dataclass(frozen=True)
class BaseQwen:
    tokenizer: Any
    model: torch.nn.Module

def load_qwen_base_for_generation(
    config: QwenTrainingConfig, *, device: int = 0
) -> BaseQwen: ...

def restore_qwen_for_inference(
    config: QwenTrainingConfig,
    *, checkpoint: str | Path,
    run_name: str,
    device: int = 0,
) -> LoadedQwen: ...
```

The Base loader must load `QwenTrainingConfig.model_path` as the original Qwen
Instruct tokenizer/model without PEFT, checkpoint restoration, added tokens, or
resized embeddings. The Student restore helper should move the existing
checkpoint restore logic out of `pipeline._restore_for_inference`; update the
existing pipeline helper to delegate to this public function so normal
`generate` behavior remains unchanged.

- [ ] **Step 4: Implement generation orchestration**

In `quality_generation.py`, define a small adapter protocol with `generate` and
`close`. Implement Base generation with `BaseQwen.model.generate` and Student
generation with `LoadedQwen.slot_model.generate`; both use
`max_new_tokens`, `do_sample=False`, and `use_cache=True`.

Process sources in configuration order but load only one local adapter at a
time. Append every `QualityResponse` immediately. For Teacher, create rows from
`final_response` without constructing an adapter. On `close`, drop references
and call `torch.cuda.empty_cache()` only when CUDA is available.

Write a readable manifest before work begins. On resume, validate it and skip
existing `(split, example_id, model_id)` keys. Use `append_jsonl` for successes
and `QualityFailure` rows.

- [ ] **Step 5: Run the core generation test and the existing generation tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_quality_evaluation.py::test_generation_reuses_teacher_and_keeps_ibd_tokens_out_of_base tests/test_student_data.py tests/test_qwen.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the generation slice**

```bash
git add src/ibd/student_data.py src/ibd/qwen.py src/ibd/pipeline.py src/ibd/quality_generation.py tests/test_quality_evaluation.py
git commit -m "feat: generate fair quality comparison responses"
```

---

### Task 3: Five-dimension anonymous LLM Judge

**Files:**
- Create: `src/ibd/quality_judge.py`
- Modify: `tests/test_quality_evaluation.py`

**Interfaces:**
- Consumes: `QualityEvalConfig`, `QualityResponse`, `QualityJudgment`, `QualityScores`.
- Produces: `QUALITY_JUDGE_RUBRIC`, `QUALITY_JUDGE_SYSTEM_PROMPT` local constants.
- Produces: `QualityJudge.score(response: QualityResponse) -> QualityJudgment`.
- Produces: `judge_quality_responses(config, responses, output_path, manifest_path, failure_path, resume, continue_on_error, backend=None) -> int`.

- [ ] **Step 1: Add one Judge protocol test**

```python
def test_quality_judge_is_anonymous_and_has_exactly_five_dimensions(history):
    backend = QualityScriptedBackend()
    response = quality_response_fixture(history, model_id="student")
    judgment = QualityJudge(backend, quality_config_fixture()).score(response)
    request = backend.calls[0]
    prompt_text = str(request["messages"])
    assert "student" not in prompt_text
    assert "checkpoint" not in prompt_text.lower()
    assert "safety" not in prompt_text.lower()
    assert judgment.overall == 4.0
    assert set(judgment.scores.model_dump()) == {
        "empathy", "relevance", "coherence", "effectiveness", "non_coerciveness"
    }
```

- [ ] **Step 2: Run the Judge test and verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_quality_evaluation.py::test_quality_judge_is_anonymous_and_has_exactly_five_dimensions -q`

Expected: FAIL because `QualityJudge` does not exist.

- [ ] **Step 3: Implement the local rubric and structured call**

Adapt the supervisor rubric descriptors for the five retained dimensions. The
system prompt must state that the Judge sees one blinded candidate, must score
dimensions independently, must cite a concrete phrase/omission/decision, and
must return JSON only. It must not contain the word `safety` or any safety
schema field.

Construct an ephemeral `AppConfig` from the quality config with role
`quality_judge`, then use the existing `StructuredCaller` for schema retry and
cache behavior. The user message contains exactly:

```text
FIXED HISTORY H:
<History.as_prompt()>

CANDIDATE NEXT SUPPORTER RESPONSE:
<response>
```

Use a cache `example_id` that includes split, example ID, and anonymous row
ordinal/model-independent response key stored in the manifest; do not add a new
hash implementation. Compute `overall` locally and persist `CallRecord` values.

- [ ] **Step 4: Add durable Judge orchestration and report writing**

Append judgments as they succeed, write failures as `QualityFailure`, validate
resume manifests, and call `build_quality_report` after processing. Refuse to
present full coverage when any expected response key lacks a judgment. Record
Judge/Teacher identity match as `true`, `false`, or `unknown` plus a warning; do
not block a match.

- [ ] **Step 5: Run the Judge and aggregation tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_quality_evaluation.py -k 'judge or report' -q`

Expected: PASS.

- [ ] **Step 6: Commit the Judge slice**

```bash
git add src/ibd/quality_judge.py src/ibd/quality.py tests/test_quality_evaluation.py
git commit -m "feat: add blinded five-dimension judge"
```

---

### Task 4: Optional human pairwise export and aggregation

**Files:**
- Create: `src/ibd/quality_human.py`
- Modify: `tests/test_quality_evaluation.py`

**Interfaces:**
- Consumes: validated `QualityResponse` rows.
- Produces: `export_human_pairs(responses, public_csv, mapping_jsonl, seed) -> dict[str, int]`.
- Produces: `summarize_human_annotations(annotation_csv, mapping_jsonl) -> dict[str, object]`.

- [ ] **Step 1: Add one round-trip human-review test**

```python
def test_human_pair_export_is_blinded_deterministic_and_summarizable(tmp_path, history):
    responses = three_model_response_fixture(history, examples=2)
    public_path = tmp_path / "pairs.csv"
    mapping_path = tmp_path / "mapping.jsonl"
    export_human_pairs(responses, public_path, mapping_path, seed=73)
    first = public_path.read_text(encoding="utf-8")
    export_human_pairs(responses, public_path, mapping_path, seed=73)
    assert public_path.read_text(encoding="utf-8") == first
    assert "student" not in first and "teacher" not in first and "base" not in first
    fill_preferences(public_path, ["A", "B", "tie", "A", "B", "tie"])
    report = summarize_human_annotations(public_path, mapping_path)
    assert sum(group["valid_votes"] for group in report["groups"].values()) == 6
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_quality_evaluation.py::test_human_pair_export_is_blinded_deterministic_and_summarizable -q`

Expected: FAIL because `quality_human` does not exist.

- [ ] **Step 3: Implement deterministic export**

Validate complete same-history response groups. Generate every unordered model
pair within each split. Sort example IDs, seeded-shuffle within the group, and
alternate orientation. Assign `pair-000001` style sequential IDs after stable
group ordering. Write public CSV columns exactly as specified in the design and
write `pair_id`, `split`, `example_id`, `model_a`, `model_b` to private JSONL.

Never copy model IDs, response sources, checkpoint fields, or generation
metadata into public columns. Use atomic overwrite for export because it is a
deterministic derivative, not an expensive append-only stage.

- [ ] **Step 4: Implement optional aggregation**

Accept `A`, `B`, or `tie` case-insensitively after trimming. Join annotations to
the private mapping, translate A/B back to model wins, and group by split and
unordered model pair. Preserve every valid duplicate `pair_id` row as a separate
reviewer vote. Report raw wins for each model, ties, valid votes, missing rows,
invalid rows, and rates over valid votes.

- [ ] **Step 5: Run the human round-trip test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_quality_evaluation.py::test_human_pair_export_is_blinded_deterministic_and_summarizable -q`

Expected: PASS.

- [ ] **Step 6: Commit the human-review slice**

```bash
git add src/ibd/quality_human.py tests/test_quality_evaluation.py
git commit -m "feat: add optional blinded human review"
```

---

### Task 5: CLI integration and example configuration

**Files:**
- Create: `src/ibd/quality_cli.py`
- Create: `configs/quality_eval.yaml`
- Modify: `src/ibd/cli.py`
- Modify: `tests/test_quality_evaluation.py`

**Interfaces:**
- Consumes all orchestration functions from Tasks 2--4.
- Produces: `register_quality_commands(commands) -> None`.
- Produces: `run_quality_command(args: Namespace) -> int`.

- [ ] **Step 1: Add one offline CLI smoke test**

Test parser registration and dispatch with monkeypatched generation adapters and
scripted Judge backend. Drive this sequence through `ibd.cli.main`:

```python
assert main(["quality-generate", "--config", config, "--traces", traces,
             "--output-dir", generated]) == 0
assert main(["quality-judge", "--config", config, "--responses",
             generated / "responses.jsonl", "--output-dir", judged]) == 0
assert main(["quality-human-export", "--config", config, "--responses",
             generated / "responses.jsonl", "--output-dir", human]) == 0
assert main(["quality-human-summarize", "--annotations", human / "pairs.csv",
             "--mapping", human / "private_mapping.jsonl",
             "--output", human / "report.json"]) == 0
```

Assert only that the four expected final artifacts exist and that the Judge
report contains separate `dev` and `diagnostic_holdout` keys. Do not duplicate
all unit assertions in this smoke test.

- [ ] **Step 2: Run the smoke test and verify parser failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_quality_evaluation.py -k cli_smoke -q`

Expected: FAIL because quality subcommands are unknown.

- [ ] **Step 3: Implement focused CLI orchestration**

`quality_cli.py` owns parser definitions and filesystem conventions:

- `quality-generate --config --traces --output-dir [--resume] [--continue-on-error] [--device]`
- `quality-judge --config --responses --output-dir [--resume] [--continue-on-error]`
- `quality-human-export --config --responses --output-dir`
- `quality-human-summarize --annotations --mapping --output`

Use fixed filenames inside output directories: `responses.jsonl`,
`responses.manifest.json`, `generation_failures.jsonl`, `judgments.jsonl`,
`judgments.manifest.json`, `judge_failures.jsonl`, `quality_report.json`,
`pairs.csv`, and `private_mapping.jsonl`.

In `cli.py`, call `register_quality_commands(commands)` from `_build_parser` and
dispatch command names beginning with `quality-` to `run_quality_command` before
loading unrelated input records. Keep all existing resumable Teacher code intact.

- [ ] **Step 4: Add the canonical example config**

Create `configs/quality_eval.yaml` with protocol v1, both splits, deterministic
generation, three model entries named `teacher`, `base`, and `student`, the
existing Qwen training config path, an example checkpoint/run name, backend env
names, a fixed `deepseek-v4-flash` Judge role, Judge seed, and human seed. Paths
are examples and must be clearly documented as user-editable.

- [ ] **Step 5: Run the smoke and existing CLI tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_quality_evaluation.py -k cli_smoke tests/test_cli.py -q`

Expected: PASS, including the user's existing resumable Teacher tests.

- [ ] **Step 6: Commit the CLI slice**

```bash
git add src/ibd/quality_cli.py src/ibd/cli.py configs/quality_eval.yaml tests/test_quality_evaluation.py
git commit -m "feat: expose generation quality workflow"
```

---

### Task 6: Documentation and proportional verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-26-generation-quality-evaluation-design.md`

**Interfaces:**
- Documents the exact CLI and artifact flow produced by Task 5.

- [ ] **Step 1: Document the workflow**

Add a `生成质量对比` section explaining:

- how to copy/edit `configs/quality_eval.yaml`;
- that Teacher responses consume no new tokens;
- that Base does not receive Student-only structure tokens;
- four representative commands in execution order;
- the five Judge dimensions and arithmetic `overall`;
- separate dev/holdout reporting;
- optional human annotations with `A`, `B`, or `tie`;
- adding another local checkpoint by adding a model registry entry.

- [ ] **Step 2: Run the focused quality suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_quality_evaluation.py -q`

Expected: PASS.

- [ ] **Step 3: Run the full offline suite once**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`

Expected: PASS. If a failure is unrelated to this feature and predates the
change, record the exact failure rather than modifying unrelated code.

- [ ] **Step 4: Check formatting and repository state**

Run: `git diff --check`

Expected: no whitespace errors. Confirm `git status --short` still shows any
pre-existing user changes not included in feature commits.

- [ ] **Step 5: Commit documentation and the approved spec clarification**

```bash
git add README.md docs/superpowers/specs/2026-08-26-generation-quality-evaluation-design.md
git commit -m "docs: explain generation quality evaluation"
```

---

## Final Acceptance

- `quality-generate` produces one Teacher/Base/Student response per selected history and can resume.
- `quality-judge` sees no model identity, uses exactly five non-safety dimensions, and emits split-specific paired reports.
- `quality-human-export` produces deterministic anonymous comparisons and a separate private mapping.
- `quality-human-summarize` is optional and correctly translates `A`/`B`/`tie` votes.
- Base generation uses the original vocabulary while Student generation uses trained IBD structure tokens.
- Core quality tests and the existing repository test suite pass without reverting the user's unrelated changes.
