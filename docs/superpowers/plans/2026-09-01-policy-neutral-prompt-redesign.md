# Policy-Neutral Prompt Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved seven-dimensional, policy-neutral Teacher, intervention, and Student prompt pipeline on the current local branch.

**Architecture:** Add one canonical runtime registry for STATE label meanings and local effects, then generate prompt appendices and counterfactual context from that registry. Update schemas and controllers to support one to three Candidates, retain structured Candidate retry with audited fallback metadata, and make effect verification reject non-local STATE changes. Use one explicit Student-family system message across training and inference.

**Tech Stack:** Python 3.11+, Pydantic v2, PyTorch, Hugging Face Transformers, pytest, YAML configuration.

## Global Constraints

- Work only on the current local branch `codex/implementation`; do not push or create a PR.
- Preserve the user's existing uncommitted edits in `src/ibd/schemas.py`, `tests/test_schemas.py`, and `configs/deepseek_teacher_phase_balanced_v4.yaml`.
- Do not restore or commit the currently deleted `AGENTS.md` or `CLAUDE.md`.
- STATE meanings and enum values come from `docs/superpowers/specs/2026-08-31-seven-dimensional-user-state-design.md`.
- Prompt wording and behavior come from `docs/superpowers/specs/2026-09-01-policy-neutral-prompt-redesign-design.md`.
- Planner returns one to three distinct strategies; it returns only `Others` for an unambiguously closing turn with no unfinished request.
- Candidate Self-disclosure may use only brief, generic, low-risk synthetic
  supporter experience and must return the focus to the seeker.
- Final Selector uses conditional fit before response quality.
- Candidate plain text triggers one schema retry; second-attempt plain text receives the approved generic metadata and an audit flag.
- `unknown`, `<MASKED>`, and the original value are never legal STATE counterfactual replacements.
- Do not modify the independent quality-judge rubric.
- Use this exact Student-family system message: `You are an AI mental-health support counselor. Provide compassionate, attentive, and context-sensitive conversational support based on the visible dialogue. Respond naturally and respect the seeker's autonomy, pace, boundaries, and expressed needs. Avoid unsupported assumptions and diagnosis.`

---

### Task 1: Canonical STATE Prompt Registry and Fixture Migration

**Files:**
- Create: `src/ibd/state_guides.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_anchors.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_export.py`
- Modify: `tests/test_interventions.py`
- Modify: `tests/test_teacher.py`
- Test: `tests/test_state_guides.py`

**Interfaces:**
- Consumes: `StateField` and `STATE_ANCHOR_FIELDS` from `ibd.schemas`.
- Produces: `StateFieldGuide`, `STATE_FIELD_GUIDES`, `render_state_label_guide()`, and `counterfactual_context(field, original_value)`.

- [ ] **Step 1: Write failing registry tests**

```python
from ibd.schemas import STATE_ANCHOR_FIELDS, StateBlackboard
from ibd.state_guides import (
    STATE_FIELD_GUIDES,
    counterfactual_context,
    render_state_label_guide,
)


def test_state_guides_cover_schema_fields_and_enum_values():
    assert tuple(STATE_FIELD_GUIDES) == STATE_ANCHOR_FIELDS
    properties = StateBlackboard.model_json_schema()["properties"]
    for field, guide in STATE_FIELD_GUIDES.items():
        assert tuple(properties[field]["enum"]) == tuple(guide.values)


def test_counterfactual_context_excludes_original_and_missing_values():
    context = counterfactual_context("advice_receptivity", "hesitant")
    assert context["allowed_replacements"] == ["closed", "open", "requested"]
    assert "unknown" not in context["allowed_replacements"]
    assert context["original_value"] == "hesitant"


def test_state_label_guide_contains_high_risk_boundaries():
    prompt = render_state_label_guide()
    assert "Absence of an advice request does not imply closed" in prompt
    assert "Wanting to act but feeling unable is not ambivalent" in prompt
    assert "terminal thanks without an unfinished request" in prompt
```

- [ ] **Step 2: Run registry tests and verify the missing-module failure**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_state_guides.py -q`

Expected: FAIL during import because `ibd.state_guides` does not exist.

- [ ] **Step 3: Implement the registry**

Create an immutable guide type and seven ordered entries:

```python
@dataclass(frozen=True)
class StateFieldGuide:
    definition: str
    values: dict[str, str]
    isolation_rule: str
    permitted_local_effects: tuple[str, ...]
    prohibited_local_effects: tuple[str, ...]


def counterfactual_context(field: StateField, original_value: str) -> dict[str, object]:
    guide = STATE_FIELD_GUIDES[field]
    return {
        "target_field": field,
        "target_field_definition": guide.definition,
        "original_value": original_value,
        "allowed_replacements": [
            value
            for value in guide.values
            if value not in {original_value, "unknown"}
        ],
        "permitted_local_effects": list(guide.permitted_local_effects),
    }
```

Populate every enum definition, exclusion boundary, isolation rule, and local effect exactly from the canonical seven-dimensional STATE specification. `render_state_label_guide()` must render all seven fields in `STATE_ANCHOR_FIELDS` order.

- [ ] **Step 4: Migrate shared and file-local fixtures from the old STATE shape**

Use this canonical fixture wherever a valid STATE is needed:

```python
VALID_STATE = {
    "dominant_emotion": "hurt_disappointment",
    "distress_level": "moderate",
    "primary_support_need": "decision_support",
    "advice_receptivity": "hesitant",
    "action_intent": "considering",
    "action_capacity": "limited",
    "continuation_intent": "engaged",
}
```

Add matching `state_evidence` to scripted analyzer outputs. Replace diagnostic uses of `readiness` with `advice_receptivity` unless the fixture specifically tests another new field.

- [ ] **Step 5: Run schema, registry, and fixture-dependent tests**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_schemas.py tests/test_state_guides.py tests/test_anchors.py tests/test_export.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the registry and fixture migration**

```bash
git add src/ibd/state_guides.py tests/conftest.py tests/test_state_guides.py tests/test_anchors.py tests/test_cli.py tests/test_export.py tests/test_interventions.py tests/test_teacher.py src/ibd/schemas.py tests/test_schemas.py
git commit -m "feat: define seven-state prompt registry"
```

---

### Task 2: Main Teacher Prompts and Shared Strategy Catalog

**Files:**
- Modify: `src/ibd/prompting.py`
- Test: `tests/test_teacher.py`

**Interfaces:**
- Consumes: `render_state_label_guide()` and `STATE_FIELD_GUIDES` from Task 1.
- Produces: approved `_ROLE_PROMPTS`, shared `ESCONV_STRATEGIES`, and resolved STATE/PLAN clamp suffixes.

- [ ] **Step 1: Replace stale prompt assertions with approved-policy tests**

```python
def test_analyzer_prompt_uses_seven_state_guide(history):
    prompt = build_messages(
        "multi_view_state_analyzer", history, MultiViewStateAnalysis
    )[0]["content"]
    assert "seven-dimensional user-state analyzer" in prompt
    assert "advice_receptivity" in prompt
    assert "unknown as a middle or low value" in prompt
    assert "at most 20 words" not in prompt


def test_planner_prompt_allows_only_meaningful_one_to_three_options(history):
    prompt = build_messages(
        "planner", history, StrategyPlanSet, context={"state": VALID_STATE}
    )[0]["content"]
    assert "between one and three" in prompt
    assert "return only Others" in prompt
    assert "merely to increase the number" in prompt
    for strategy in ESCONV_STRATEGIES:
        assert f"{strategy}:" in prompt


def test_candidate_prompt_has_bounded_synthetic_self_disclosure(history):
    prompt = candidate_prompt(history, strategy="Self-disclosure")
    assert "natural, contextually appropriate" in prompt
    assert "functionally from other candidates" not in prompt
    assert "brief, generic, low-risk synthetic supporter experience" in prompt
    assert "I went through something similar" in prompt
    assert "professional qualifications" in prompt


def test_final_selector_uses_conditional_fit_then_quality(history):
    prompt = selector_prompt(history)
    assert "Stage 1 — Conditional fit" in prompt
    assert "Stage 2 — Response quality" in prompt
    assert "STATE.primary_need" not in prompt
    assert "concrete, autonomy-preserving" not in prompt
```

- [ ] **Step 2: Run prompt tests and verify failures against old wording**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_teacher.py -q`

Expected: FAIL on the new approved wording and removed old wording.

- [ ] **Step 3: Replace the strategy catalog and all eight role prompts**

Copy the exact approved catalog and prompt text from the design specification. Append the full catalog to Planner, only the assigned strategy to Candidate, and the generated STATE label guide to the Analyzer. Replace Final Selector with version B. Replace STATE/PLAN verifier and Safety prompts with the approved text.

- [ ] **Step 4: Replace dynamic clamp suffixes**

For a STATE clamp, resolve its field guide and append the authoritative experimental condition, six-field stability, and permitted-local-effect text. For a fixed PLAN, append the authoritative strategy/goal/act text and explain precedence when both clamps exist.

- [ ] **Step 5: Run prompt tests**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_teacher.py -q`

Expected: prompt construction tests PASS; controller-cardinality tests may remain pending Task 3.

- [ ] **Step 6: Commit prompt changes**

```bash
git add src/ibd/prompting.py tests/test_teacher.py
git commit -m "feat: replace policy-biased teacher prompts"
```

---

### Task 3: Variable Candidate Cardinality and Audited Plain-Text Fallback

**Files:**
- Modify: `src/ibd/schemas.py`
- Modify: `src/ibd/teacher.py`
- Modify: `src/ibd/backend.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_schemas.py`
- Modify: `tests/test_teacher.py`

**Interfaces:**
- Produces: `StrategyPlanSet.strategies` and `TeacherTrace.candidates` with lengths 1-3; `CallRecord.metadata_fallback: bool = False`; `_PlainTextCandidateBackend.fallback_used`.

- [ ] **Step 1: Add failing cardinality and retry tests**

```python
@pytest.mark.parametrize("strategies", [
    ["Others"],
    ["Question", "Reflection of feelings"],
    ["Question", "Information", "Providing Suggestions"],
])
def test_planner_accepts_one_to_three_distinct_strategies(strategies):
    assert StrategyPlanSet(strategies=strategies).strategies == strategies


def test_one_strategy_generates_one_candidate_and_one_selector_call(history, app_config):
    backend = ScriptedBackend(planner_strategies=["Others"])
    trace = TeacherRunner(backend, app_config).run("closing", history)
    assert len(trace.candidates) == 1
    assert [call["role"] for call in backend.calls].count("candidate") == 1
    assert [call["role"] for call in backend.calls].count("final_selector") == 1


def test_plain_text_candidate_retries_then_uses_audited_fallback(history):
    backend = AlwaysPlainCandidateBackend()
    trace = TeacherRunner(backend, candidate_plain_config).run("plain", history)
    candidate_records = [r for r in trace.call_records if r.role == "candidate"]
    assert len(backend.candidate_calls) == 6
    assert all(r.metadata_fallback for r in candidate_records if r.parsed)
    assert all(c.response_goal == "Support the seeker's immediate goal" for c in trace.candidates)
```

- [ ] **Step 2: Run targeted tests and verify fixed-three/immediate-wrap failures**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_schemas.py tests/test_teacher.py -q`

Expected: FAIL because schemas require three, the loop uses `range(1, 4)`, and plain text is wrapped before retry.

- [ ] **Step 3: Relax schemas and drive generation from Planner output**

Change both annotated list fields to `Field(min_length=1, max_length=3)`. Generate Candidates with:

```python
candidates = [
    self._generate_candidate(
        example_id=example_id,
        history=history,
        state=state,
        strategy=strategy,
        local_index=index,
        records=records,
        clamped_state_field=clamped_state_field,
    )
    for index, strategy in enumerate(plan.strategies, start=1)
]
```

Keep `S1`, `S2`, and `S3` IDs and preserve the existing final-selection provenance checks.

- [ ] **Step 4: Make plain Candidate text retry before fallback**

Track backend completion count inside the Candidate adapter. Return first-attempt plain text unchanged so `StructuredCaller` records a schema failure and sends its single retry. On second-attempt plain text, wrap it with frozen IDs and the approved generic goal/act. Expose `fallback_used` and set `CallRecord.metadata_fallback=True` on the successful wrapped attempt.

- [ ] **Step 5: Run schema and Teacher tests**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_schemas.py tests/test_teacher.py -q`

Expected: PASS.

- [ ] **Step 6: Commit cardinality and fallback behavior**

```bash
git add src/ibd/schemas.py src/ibd/teacher.py src/ibd/backend.py tests/conftest.py tests/test_schemas.py tests/test_teacher.py
git commit -m "feat: support variable teacher candidates"
```

---

### Task 4: Enum STATE Counterfactuals and Diagnostic Mask Boundary

**Files:**
- Modify: `src/ibd/interventions.py`
- Modify: `src/ibd/teacher.py`
- Modify: `src/ibd/pipeline.py`
- Modify: `src/ibd/anchors.py`
- Modify: `tests/test_interventions.py`
- Modify: `tests/test_teacher.py`
- Modify: `tests/test_anchors.py`

**Interfaces:**
- Consumes: `counterfactual_context()` from Task 1.
- Produces: enum-valid `generate_state_counterfactual(history: History, state: StateBlackboard, target_field: StateField, *, example_id: str) -> str`; diagnostic masked anchor payloads without constructing invalid `StateBlackboard` values.

- [ ] **Step 1: Add failing enum counterfactual and mask-boundary tests**

```python
def test_state_counterfactual_context_uses_target_enum(history, app_config):
    runner = TeacherRunner(ScriptedBackend(), app_config)
    replacement = runner.generate_state_counterfactual(
        history, StateBlackboard(**VALID_STATE), "advice_receptivity", example_id="cf"
    )
    assert replacement in {"closed", "open", "requested"}


def test_diagnostic_mask_exists_only_in_serialized_anchor_payload():
    state = StateBlackboard(**VALID_STATE)
    payload = masked_state_anchor_payload(state, "advice_receptivity")
    assert payload["advice_receptivity"] == "<MASKED>"
    with pytest.raises(ValidationError):
        StateBlackboard(**payload)
```

- [ ] **Step 2: Run tests and verify old-field/free-text failures**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_interventions.py tests/test_teacher.py tests/test_anchors.py -q`

Expected: FAIL on old `StateField`, old target-dimension mapping, and invalid masked `StateBlackboard` construction.

- [ ] **Step 3: Replace duplicate old StateField and target mappings**

Import `StateField` from `schemas.py`. Use this audit mapping:

```python
STATE_TARGET_DIMENSIONS = {
    "dominant_emotion": "emotion",
    "distress_level": "timing",
    "primary_support_need": "need",
    "advice_receptivity": "autonomy",
    "action_intent": "intent",
    "action_capacity": "effectiveness",
    "continuation_intent": "timing",
}
```

Build generator context from `counterfactual_context()` plus the full STATE. Validate the returned string by replacing exactly the target field through `StateBlackboard.model_validate()`.

- [ ] **Step 4: Move diagnostic masking to anchor serialization**

Add:

```python
def masked_state_anchor_payload(state: StateBlackboard, field: StateField) -> dict[str, str]:
    payload = state_anchor_payload(state)
    payload[field] = MASKED_STATE_VALUE
    return payload
```

Store diagnostic masked payload text directly for anchor encoding rather than storing an invalid mutated `StateBlackboard`.

- [ ] **Step 5: Make PLAN intervention eligibility cardinality-aware**

Return `plan_no_alternative` from the intervention shape check when fewer than two unselected Candidates exist. Avoid calling `select_counterfactual_plan()` before this eligibility check in both intervention construction and audit preparation.

- [ ] **Step 6: Run counterfactual, anchor, and intervention tests**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_interventions.py tests/test_teacher.py tests/test_anchors.py tests/test_evaluation.py -q`

Expected: PASS.

- [ ] **Step 7: Commit enum counterfactual migration**

```bash
git add src/ibd/interventions.py src/ibd/teacher.py src/ibd/pipeline.py src/ibd/anchors.py tests/test_interventions.py tests/test_teacher.py tests/test_anchors.py tests/test_evaluation.py
git commit -m "feat: localize enum state counterfactuals"
```

---

### Task 5: Strict STATE Effect Localization and Updated Audits

**Files:**
- Modify: `src/ibd/interventions.py`
- Modify: `src/ibd/pipeline.py`
- Modify: `src/ibd/schemas.py`
- Modify: `src/ibd/evaluation.py`
- Modify: `src/ibd/export.py`
- Modify: `tests/test_interventions.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_export.py`

**Interfaces:**
- Produces: `StateEffectVerdict(condition_a_fit, condition_b_fit, target_effect_present, localized_effect, affected_non_target_fields, evidence)`, `EffectVerification.affected_non_target_fields`, and `_combine_state_effect_verdicts(first: StateEffectVerdict, second: StateEffectVerdict) -> EffectVerification`.

- [ ] **Step 1: Add failing strict-localization tests**

```python
def test_state_effect_verdict_records_non_target_fields():
    verdict = StateEffectVerdict(
        condition_a_fit=True,
        condition_b_fit=True,
        target_effect_present=True,
        localized_effect=False,
        affected_non_target_fields=["action_capacity"],
        evidence="Advice posture changed, but step burden also changed.",
    )
    assert verdict.affected_non_target_fields == ["action_capacity"]


def test_state_effect_rejects_non_local_change():
    first = StateEffectVerdict(
        condition_a_fit=True,
        condition_b_fit=True,
        target_effect_present=True,
        localized_effect=False,
        affected_non_target_fields=["action_capacity"],
        evidence="Changing advice posture also changed the proposed step burden.",
    )
    second = StateEffectVerdict(
        condition_a_fit=True,
        condition_b_fit=True,
        target_effect_present=True,
        localized_effect=False,
        affected_non_target_fields=["action_capacity"],
        evidence="The swapped bundles show the same non-target capacity effect.",
    )
    result = _combine_state_effect_verdicts(first, second)
    assert result.passed is False
    assert result.reason == "non_localized_effect"
```

- [ ] **Step 2: Run verifier tests and verify old verdict-shape failures**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_interventions.py tests/test_evaluation.py tests/test_export.py -q`

Expected: FAIL because the old verdict exposes `field_effect_present` and `affected_dimensions`.

- [ ] **Step 3: Implement the new verdict and retention gate**

Extract AB/BA combination into `_combine_state_effect_verdicts`. Require agreement for condition fits, target effect, localization, and swapped non-target field sets. Reject in this order: bidirectional disagreement, condition mismatch/no target effect, then non-localized effect. Add `non_localized_effect` to exclusion and audit reason literals.

- [ ] **Step 4: Update intervention and export audit fields**

Replace `affected_dimensions` with `affected_non_target_fields` wherever the data refers to STATE localization. Preserve `target_dimension` only as the existing coarse experiment-report field. Update aggregation to report per-target-field attempts, retained counts, and non-local rejection counts.

- [ ] **Step 5: Run verifier/export tests**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_interventions.py tests/test_evaluation.py tests/test_export.py -q`

Expected: PASS.

- [ ] **Step 6: Commit strict localization**

```bash
git add src/ibd/interventions.py src/ibd/pipeline.py src/ibd/schemas.py src/ibd/evaluation.py src/ibd/export.py tests/test_interventions.py tests/test_evaluation.py tests/test_export.py
git commit -m "feat: reject non-local state effects"
```

---

### Task 6: Explicit Student-Family System Prompt Parity

**Files:**
- Modify: `src/ibd/student_data.py`
- Modify: `src/ibd/quality_generation.py`
- Modify: `tests/test_student_data.py`
- Modify: `tests/test_quality_generation.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `STUDENT_SYSTEM_PROMPT` and `history_messages(history: History | Mapping[str, Any]) -> list[dict[str, str]]` used by all chat-template paths.

- [ ] **Step 1: Add failing system-message parity tests**

```python
def test_history_messages_prepends_approved_system_prompt(history):
    messages = history_messages(history)
    assert messages[0] == {"role": "system", "content": STUDENT_SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"


def test_approved_system_prompt_has_no_strategy_or_lived_experience_rule():
    assert "Question" not in STUDENT_SYSTEM_PROMPT
    assert "Providing Suggestions" not in STUDENT_SYSTEM_PROMPT
    assert "personal or lived experience" not in STUDENT_SYSTEM_PROMPT
    assert STUDENT_SYSTEM_PROMPT.endswith("Avoid unsupported assumptions and diagnosis.")
```

- [ ] **Step 2: Run Student tests and verify default-template failure**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_student_data.py tests/test_quality_generation.py -q`

Expected: FAIL because histories currently start with `user` and rely on Qwen's implicit default system message.

- [ ] **Step 3: Add the explicit constant and use it in every encoder**

```python
STUDENT_SYSTEM_PROMPT = (
    "You are an AI mental-health support counselor. Provide compassionate, "
    "attentive, and context-sensitive conversational support based on the visible "
    "dialogue. Respond naturally and respect the seeker's autonomy, pace, boundaries, "
    "and expressed needs. Avoid unsupported assumptions and diagnosis."
)
```

Prepend it in `history_messages()`. Because Base, IBD Student, SFT control, and visible-SFT adapters all call the same encoding helpers, verify rather than duplicate prompt insertion in `quality_generation.py`.

- [ ] **Step 4: Run Student and quality-generation tests**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_student_data.py tests/test_quality_generation.py tests/test_cli.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Student prompt parity**

```bash
git add src/ibd/student_data.py src/ibd/quality_generation.py tests/test_student_data.py tests/test_quality_generation.py tests/test_cli.py
git commit -m "feat: unify student support system prompt"
```

---

### Task 7: Protocol Versioning, Documentation, and Full Regression

**Files:**
- Modify: `configs/deepseek_teacher.yaml`
- Modify: `configs/deepseek_teacher_phase_balanced_v4.yaml`
- Modify: `README.md`
- Modify: prompt/schema/controller tests as required by full-suite failures

**Interfaces:**
- Produces: a new cache namespace through protocol-version changes and accurate CLI/documentation examples using current STATE fields.

- [ ] **Step 1: Add protocol and stale-documentation assertions**

```python
def test_teacher_configs_use_synthetic_self_disclosure_protocol():
    for path in TEACHER_CONFIGS:
        config = AppConfig.from_yaml(path)
        assert config.protocol_version == (
            "qwen25-socialsim-seven-state-policy-neutral-"
            "v2-synthetic-self-disclosure"
        )
```

Add an exact-string documentation test or repository check that rejects old diagnostic field examples such as `--diagnostic-state-field readiness`.

- [ ] **Step 2: Run the full suite before final documentation fixes**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest -q`

Expected: any remaining failures identify stale old-field fixtures, fixed-three assumptions, or prompt assertions; no network or GPU tests should run.

- [ ] **Step 3: Bump protocols and update documentation**

Set both Teacher configs to:

```yaml
protocol_version: qwen25-socialsim-seven-state-policy-neutral-v2-synthetic-self-disclosure
```

Update README architecture from fixed three Candidates to one to three, replace old STATE field examples and visible-SFT marker order, document one-Candidate PLAN-intervention exclusion, and state that new Teacher/intervention/anchor artifacts must be generated in a new directory.

- [ ] **Step 4: Run static prompt-bias scans**

Run:

```bash
rg -n 'STATE\.(primary_need|support_goal|readiness|main_constraint)|functionally distinct from other candidates|fill three strategy slots|prefer a concrete' src tests README.md
```

Expected: no active prompt or documentation matches. Tests may contain these phrases only inside explicit negative assertions.

- [ ] **Step 5: Run the complete test suite**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest -q`

Expected: PASS with zero failures.

- [ ] **Step 6: Inspect the final diff and ensure unrelated changes remain untouched**

Run:

```bash
git diff --check
git status --short
git diff --stat HEAD~6..HEAD
```

Expected: no whitespace errors; deleted `AGENTS.md`/`CLAUDE.md` remain uncommitted unless they were already user-owned changes; no generated artifacts, caches, model files, or quality-judge changes appear.

- [ ] **Step 7: Commit protocol and documentation updates**

```bash
git add configs/deepseek_teacher.yaml configs/deepseek_teacher_phase_balanced_v4.yaml README.md tests
git commit -m "docs: finalize policy-neutral teacher protocol"
```

## Final Verification

- [ ] Run `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest -q` and record the passing test count.
- [ ] Run `git diff --check` and verify clean output.
- [ ] Review all commits after `3e69b78` and confirm each contains only its intended task.
- [ ] Confirm the working tree retains only user-owned unrelated changes, if any.
