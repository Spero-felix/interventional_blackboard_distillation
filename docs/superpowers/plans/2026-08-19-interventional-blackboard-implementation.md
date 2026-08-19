# Interventional Blackboard Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, testable implementation of the four-expert Teacher, STATE/PLAN interventions, Stage D margin-pair construction, two-token Student adapter, training losses, and evaluation metrics.

**Architecture:** The project is self-contained and does not import `supervisor_demo`. Typed Pydantic traces isolate Student-eligible data from safety audit fields. Real model calls use an OpenAI-compatible backend protocol; tests use a scripted fake backend and require no network.

**Tech Stack:** Python 3.11, Pydantic 2, PyYAML 6, OpenAI client, optional PyTorch 2.6, pytest 8.

## Global Constraints

- Four mutually invisible experts: emotion, need, relationship, intent.
- No risk expert and no pre-generation risk route.
- Normal Teacher trace: 14 logical calls; one gate repair: at most 16.
- Safety critique is Teacher-only and never becomes a Student input, slot target, intervention, or margin negative.
- Student latent names are exactly `STATE` and `PLAN`.
- Stage D uses frozen offline `(H, chosen, rejected)` pairs, chosen SFT, a length-normalized hinge margin, and low-weight Stage B/C replay.
- Tests must run without model downloads or network access.

---

### Task 1: Package foundation and machine contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/ibd/__init__.py`
- Create: `src/ibd/schemas.py`
- Create: `src/ibd/config.py`
- Create: `src/ibd/hashing.py`
- Create: `src/ibd/storage.py`
- Test: `tests/test_schemas.py`
- Test: `tests/test_hashing_storage.py`

**Interfaces:**
- Produces: `History`, `StateBlackboard`, `SupportPlan`, `TeacherTrace`, `InterventionRecord`, `MarginPair`, `AppConfig`, `protocol_hash`, durable JSONL helpers.

- [ ] Write tests that reject histories not ending in seeker, cross-domain expert fields, unsafe margin pairs, and invalid intervention function names.
- [ ] Run `pytest tests/test_schemas.py tests/test_hashing_storage.py -q`; expect import failure.
- [ ] Implement Pydantic contracts, canonical JSON SHA-256, and flush/fsync JSONL storage.
- [ ] Re-run the two test files; expect all pass.

```python
class MarginPair(BaseModel):
    chosen: str
    rejected_candidate_id: str
    rejected: str
    defect_dimension: NonSafetyDimension
    defect_evidence: str
    order_swap_verified: bool
    safety_filter_passed: bool
```

### Task 2: Backend and Teacher controller

**Files:**
- Create: `src/ibd/backend.py`
- Create: `src/ibd/prompting.py`
- Create: `src/ibd/teacher.py`
- Test: `tests/conftest.py`
- Test: `tests/test_teacher.py`

**Interfaces:**
- Consumes: Task 1 config and trace schemas.
- Produces: `LLMBackend`, `StructuredCaller`, `TeacherRunner.run`, `TeacherRunner.rerun_downstream`.

- [ ] Write FakeBackend tests for the exact 14-role logical order and the 16-call repair cap.
- [ ] Run `pytest tests/test_teacher.py -q`; expect missing module failure.
- [ ] Implement structured JSON calls, one Schema retry, four independent experts, three seeded candidates, three critics, final integration, blind gate, and one repair.
- [ ] Re-run `pytest tests/test_teacher.py -q`; expect all pass and no `risk_*` role.

```python
NORMAL_ROLES = (
    "emotion_expert", "need_expert", "relationship_expert", "intent_expert",
    "state_integrator", "planner",
    "candidate_1", "candidate_2", "candidate_3",
    "emotion_critic", "effectiveness_critic", "safety_critic",
    "final_integrator", "quality_gate",
)
```

### Task 3: STATE/PLAN interventions and margin pairs

**Files:**
- Create: `src/ibd/interventions.py`
- Test: `tests/test_interventions.py`

**Interfaces:**
- Produces: `mutate_state`, `mutate_plan`, `InterventionBuilder.build`, `MarginPairBuilder.build`.

- [ ] Write tests proving exactly one STATE/PLAN field changes and accepted margin pairs use two order-swap verifier calls.
- [ ] Run `pytest tests/test_interventions.py -q`; expect missing module failure.
- [ ] Implement deterministic downgrade/reorder mutations, downstream reruns, strongest localized non-safety candidate selection, and A/B verification.
- [ ] Re-run the test file; expect unsafe candidates excluded and verifier call count equal to two.

```python
def mutate_plan(plan: SupportPlan) -> tuple[SupportPlan, Mutation]:
    updated = plan.model_copy(deep=True)
    updated.response_acts[0], updated.response_acts[1] = (
        updated.response_acts[1], updated.response_acts[0]
    )
    return updated, Mutation(
        function="PLAN", operation="reorder", changed_field_count=1
    )
```

### Task 4: Student data allowlists

**Files:**
- Create: `src/ibd/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Produces: SFT, slot, intervention, and margin JSONL exporters.

- [ ] Write tests asserting margin rows contain only `example_id`, `prompt`, `chosen`, `rejected` and contain no safety audit text.
- [ ] Run `pytest tests/test_export.py -q`; expect missing module failure.
- [ ] Implement exports by constructing allowlisted dictionaries rather than excluding fields from trace dumps.
- [ ] Re-run the test file; expect test-split exports rejected and all allowlists exact.

### Task 5: Two-token Student adapter and losses

**Files:**
- Create: `src/ibd/model.py`
- Create: `src/ibd/training.py`
- Test: `tests/test_model_training.py`

**Interfaces:**
- Produces: `LatentBlackboardCausalLM`, response-only token log-probabilities, length-normalized scores, hinge margin, and Stage D replay composition.

- [ ] Write torch tests for response masking, length normalization, zero hinge above margin, and exactly two latent parameters.
- [ ] Run `pytest tests/test_model_training.py -q`; expect missing module failure.
- [ ] Implement the adapter and losses; clamping supports only `STATE` or `PLAN`.
- [ ] Re-run the test file; expect all pass.

```python
def margin_alignment_loss(chosen, rejected, margin: float):
    return torch.relu(margin - chosen + rejected).mean()
```

### Task 6: Evaluation metrics

**Files:**
- Create: `src/ibd/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `retention`, `functional_fidelity_matrix`, `matrix_alignment`, `pair_accuracy`, `mean_rank_gap`.

- [ ] Write tests for retention denominator, strict PairAcc ties, STATE/PLAN-only matrices, sign agreement, Spearman correlation, and normalized L1 distance.
- [ ] Run `pytest tests/test_evaluation.py -q`; expect missing module failure.
- [ ] Implement metrics with explicit undefined-case errors.
- [ ] Re-run the test file; expect all pass and `CRITIC` matrix rows rejected.

### Task 7: CLI, demo configuration, and docs

**Files:**
- Create: `src/ibd/cli.py`
- Create: `configs/demo.yaml`
- Create: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces commands: `ibd protocol-hash`, `ibd validate-trace`, `ibd run-teacher`, `ibd export-student`.

- [ ] Write CLI smoke tests for a 64-character protocol hash and trace validation.
- [ ] Run `pytest tests/test_cli.py -q`; expect missing module failure.
- [ ] Implement commands and document install, FakeBackend tests, real API configuration, 14/16/18 call accounting, four stages, and research non-claims.
- [ ] Run `pytest -q`; expect the full suite to pass without network.
- [ ] Run `python -m ibd.cli protocol-hash --config configs/demo.yaml`; expect one lowercase SHA-256 line.

## Final Verification

- [ ] Run `pytest -q` and record the exact pass count.
- [ ] Run `python -m compileall -q src`.
- [ ] Run `rg -n "RISK|CRITIC|DELIB|student-in-the-loop" src`; any match must be a rejection guard, never a Student parameter or training path.
- [ ] Confirm the final project contains no import from `supervisor_demo`.
