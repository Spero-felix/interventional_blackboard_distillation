# Others Dialogue Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the catch-all `Others` prompt semantics with the approved dialogue-management definition and make Planner, Candidate Generator, and Final Selector apply the same boundary.

**Architecture:** Keep the canonical schema label `Others` unchanged. Define its positive meaning once in `src/ibd/prompting.py`, expose the explanatory display name only in prompts, add role-specific Planner and Final Selector eligibility rules in `_resolved_prompt`, and bump the active Teacher protocol version so old cached calls cannot mask the prompt change.

**Tech Stack:** Python 3.11+, Pydantic v2 structured schemas, pytest, YAML configuration, GitNexus CLI.

## Global Constraints

- Keep `Others` as the canonical code, schema, and data label.
- Use `Others (Dialogue Management and Social Courtesy)` as the explanatory prompt display name.
- `Others` manages the conversational interaction itself; it does not address the seeker's situation, emotions, beliefs, decisions, or actions.
- When another named strategy clearly describes the primary response act, that more specific strategy takes precedence.
- Do not add the proposed deletion heuristic.
- Do not add platform or data-collection noise filtering.
- Do not change the eight-strategy catalog, Pydantic schemas, Teacher call count, or Final Selector output contract.
- Do not modify the historical `configs/deepseek_teacher_phase_balanced_v4.yaml` configuration or existing v4 artifacts.
- Preserve and never stage the user's current uncommitted changes in `configs/deepseek_teacher_phase_balanced_v4.yaml`, `src/ibd/schemas.py`, and `tests/test_schemas.py`.
- Execute from an isolated worktree created from the committed branch state so those uncommitted changes cannot enter either implementation commit.
- Make no live model or network calls; verification is limited to prompt construction, configuration loading, unit tests, and GitNexus checks.

## File Structure

- Modify `src/ibd/prompting.py`: own the positive `Others` definition, prompt display name, and role-specific Planner/Candidate/Final Selector boundary text.
- Modify `tests/test_teacher.py`: verify the three Teacher roles receive and enforce the approved `Others` semantics without changing their schemas.
- Modify `configs/deepseek_teacher.yaml`: bump only the active Teacher protocol version to invalidate the config-derived cache namespace.
- Modify `tests/test_config.py`: pin the new active Teacher protocol version and retain the existing one-candidate-role assertion.

---

### Task 1: Define and expose the Others prompt boundary

**Files:**
- Modify: `src/ibd/prompting.py:12-21,177-208`
- Test: `tests/test_teacher.py:8-129`

**Interfaces:**
- Consumes: `build_messages(role: str, history: History, response_model: type[BaseModel], *, context: dict[str, Any] | None = None) -> list[dict[str, str]]`.
- Produces: unchanged canonical label `Others`; module constant `OTHERS_DISPLAY_NAME: str`; role-resolved prompt text for `planner`, `candidate`, and `final_selector`.
- Preserves: `StrategyPlanSet`, `Candidate`, and `FinalSelectionDecision` JSON schemas and all Teacher controller methods.

- [ ] **Step 1: Re-run impact analysis and report the blast radius before editing**

Run:

```bash
node .gitnexus/run.cjs impact _resolved_prompt --direction upstream --include-tests --depth 3
node .gitnexus/run.cjs impact ESCONV_STRATEGIES --direction upstream --include-tests --depth 3
```

Expected: `_resolved_prompt` reports CRITICAL risk because `build_messages` feeds normal generation, fixed-PLAN generation, state counterfactual generation, intervention verification, and Final Selector flows; `ESCONV_STRATEGIES` reports LOW risk. Warn the user about the CRITICAL shared-entry-point risk before editing and keep the implementation guarded by exact role checks.

- [ ] **Step 2: Write failing prompt-construction tests for all three roles**

In `tests/test_teacher.py`, extend the schema import and add these tests near the existing prompt tests:

```python
from ibd.schemas import Candidate, FinalSelectionDecision, StrategyPlanSet


def test_planner_prompt_limits_others_to_dialogue_management(history):
    prompt = build_messages(
        "planner",
        history,
        StrategyPlanSet,
        context={"state": {}},
    )[0]["content"]

    assert "Others (Dialogue Management and Social Courtesy)" in prompt
    assert "manage the conversational interaction itself" in prompt
    assert "greetings, brief social acknowledgments, responses to gratitude" in prompt
    assert "Never use Others as a fallback or merely to fill three strategy slots." in prompt


def test_others_candidate_prompt_uses_only_the_positive_assigned_definition(history):
    prompt = build_messages(
        "candidate",
        history,
        Candidate,
        context={
            "state": {},
            "candidate_id": "1",
            "strategy_id": "S1",
            "strategy": "Others",
        },
    )[0]["content"]

    assert "# Assigned Strategy" in prompt
    assert "Others (Dialogue Management and Social Courtesy)" in prompt
    assert "manage the conversational interaction itself" in prompt
    assert "seeker's situation, emotions, beliefs, decisions, or actions" in prompt
    assert '"const": "Others"' in prompt
    assert "Providing Suggestions:" not in prompt


def test_final_selector_prompt_gates_others_by_primary_function(history):
    prompt = build_messages(
        "final_selector",
        history,
        FinalSelectionDecision,
        context={"state": {}, "candidates": []},
    )[0]["content"]

    assert "Others (Dialogue Management and Social Courtesy)" in prompt
    assert "Select Others only when" in prompt
    assert "latest seeker turn" in prompt
    assert "that more specific strategy takes precedence over Others" in prompt
```

- [ ] **Step 3: Run the new tests to verify the red state**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest \
  tests/test_teacher.py::test_planner_prompt_limits_others_to_dialogue_management \
  tests/test_teacher.py::test_others_candidate_prompt_uses_only_the_positive_assigned_definition \
  tests/test_teacher.py::test_final_selector_prompt_gates_others_by_primary_function -v
```

Expected: all three tests FAIL because the current prompt still defines `Others` as a seven-class fallback and exposes no Planner or Final Selector boundary.

- [ ] **Step 4: Replace the fallback catalog entry with the approved positive definition**

In `src/ibd/prompting.py`, add the display-name constant immediately before `ESCONV_STRATEGIES` and replace only the `Others` value:

```python
OTHERS_DISPLAY_NAME = "Others (Dialogue Management and Social Courtesy)"

ESCONV_STRATEGIES = {
    # Keep the seven existing entries byte-for-byte unchanged.
    "Others": (
        "Use this strategy only when the primary function of the response is to "
        "manage the conversational interaction itself rather than to address the "
        "seeker's situation, emotions, beliefs, decisions, or actions. This includes "
        "greetings, brief social acknowledgments, responses to gratitude, "
        "conversational pacing or transitions, and appropriate closing or farewell "
        "messages. Do not select Others if the response primarily asks for information, "
        "restates the seeker's meaning, reflects the seeker's feelings, shares personal "
        "experience, provides affirmation or reassurance, offers factual information, "
        "or gives suggestions. When another strategy clearly describes the primary "
        "response act, that more specific strategy takes precedence over Others."
    ),
}
```

The comment in this plan is an editing instruction: do not insert `# Keep the seven...` into the source dictionary.

- [ ] **Step 5: Add exact Planner and Final Selector eligibility sections and the Candidate display heading**

At the start of `_resolved_prompt`, immediately after `prompt = _ROLE_PROMPTS[role]`, append the boundary only for the two roles that compare or choose strategies:

```python
    if role == "planner":
        prompt += (
            "\n\n# Others Strategy Boundary\n"
            f"{OTHERS_DISPLAY_NAME}: {ESCONV_STRATEGIES['Others']}\n"
            "Include Others only when the latest seeker turn primarily calls for "
            "greeting, brief social acknowledgment, a response to gratitude, "
            "conversational pacing or transition, or closing or farewell. "
            "Never use Others as a fallback or merely to fill three strategy slots."
        )
    elif role == "final_selector":
        prompt += (
            "\n\n# Others Strategy Boundary\n"
            f"{OTHERS_DISPLAY_NAME}: {ESCONV_STRATEGIES['Others']}\n"
            "Select Others only when the actual response's primary function and the "
            "latest seeker turn call for managing the interaction itself. When another "
            "candidate clearly realizes a named substantive support strategy that fits "
            "the seeker's current need, that more specific strategy takes precedence "
            "over Others."
        )
```

Within the existing `if role == "candidate":` block, preserve validation and fixed-PLAN behavior but change the assigned-strategy heading:

```python
        display_name = OTHERS_DISPLAY_NAME if strategy == "Others" else strategy
        prompt += (
            f"\n\n# Assigned Strategy\n"
            f"{display_name}: {ESCONV_STRATEGIES[strategy]}"
        )
```

Do not append the full eight-strategy catalog to Candidate prompts; the existing single-strategy isolation remains intact.

- [ ] **Step 6: Run the new tests to verify the green state**

Run the same three-node pytest command from Step 3.

Expected: `3 passed`.

- [ ] **Step 7: Run existing prompt and Teacher-path regression tests**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_teacher.py -q
```

Expected: all `tests/test_teacher.py` tests PASS. In particular, the existing assigned-strategy isolation test, Final Selector schema test, verbatim candidate selection test, fixed-PLAN test, and deterministic selector-presentation test remain green.

- [ ] **Step 8: Inspect the changed execution-flow scope before committing**

Run:

```bash
node .gitnexus/run.cjs detect_changes --scope compare --base-ref main
git diff --check
git diff -- src/ibd/prompting.py tests/test_teacher.py
```

Expected: GitNexus reports changes centered on `_resolved_prompt`/`build_messages` and the known Teacher prompt-consuming flows. The source diff changes no schemas, Teacher controller methods, candidate ordering, or output contracts.

- [ ] **Step 9: Commit the prompt behavior and regression tests**

```bash
git add src/ibd/prompting.py tests/test_teacher.py
git commit -m "feat: define Others as dialogue management"
```

Expected: the commit contains exactly the two listed files.

---

### Task 2: Isolate the new prompts from v4 Teacher caches

**Files:**
- Modify: `configs/deepseek_teacher.yaml:1`
- Test: `tests/test_config.py:12-24`

**Interfaces:**
- Consumes: `AppConfig.from_yaml(path: str | Path) -> AppConfig` and `StructuredCaller`'s existing config-derived `_cache_namespace`.
- Produces: active Teacher protocol version `qwen25-socialsim-single-strategy-v5-others-dialogue-management`.
- Preserves: role model assignments, temperatures, token limits, backend directory, schema retry count, historical phase-balanced v4 configuration, and artifact paths.

- [ ] **Step 1: Write the failing active-protocol assertion**

Extend the existing `test_deepseek_teacher_uses_one_candidate_role` in `tests/test_config.py`:

```python
    assert (
        config.protocol_version
        == "qwen25-socialsim-single-strategy-v5-others-dialogue-management"
    )
    candidate_roles = {role for role in config.roles if role.startswith("candidate")}
    assert candidate_roles == {"candidate"}
```

- [ ] **Step 2: Run the exact config test to verify the red state**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest \
  tests/test_config.py::test_deepseek_teacher_uses_one_candidate_role -v
```

Expected: FAIL because the active config still reports `qwen25-socialsim-single-strategy-v4`.

- [ ] **Step 3: Bump only the active Teacher protocol version**

Change the first line of `configs/deepseek_teacher.yaml` to:

```yaml
protocol_version: qwen25-socialsim-single-strategy-v5-others-dialogue-management
```

Do not change `cache_dir`: `StructuredCaller` already includes the complete config, including `protocol_version`, in `_cache_namespace`, so the new value produces different cache keys inside the existing directory. Do not edit `configs/deepseek_teacher_phase_balanced_v4.yaml`; it remains the reproducible historical v4 configuration.

- [ ] **Step 4: Re-run the exact config test to verify the green state**

Run the same single-node pytest command from Step 2.

Expected: `1 passed`.

- [ ] **Step 5: Run the feature-level regression suite**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest \
  tests/test_teacher.py \
  tests/test_interventions.py \
  tests/test_backend.py \
  tests/test_config.py::test_deepseek_teacher_uses_one_candidate_role -q
```

Expected: all selected tests PASS, proving prompt construction, normal and counterfactual Teacher paths, cache-key behavior, and active config loading remain compatible.

- [ ] **Step 6: Run the complete repository test suite**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest -q
```

Expected: PASS with zero failures in the isolated worktree. If the clean baseline contains a pre-existing failure unrelated to these four files, record the exact failing node and compare it against the pre-change baseline; do not claim the full suite passes.

- [ ] **Step 7: Verify the final scope before committing**

Run:

```bash
node .gitnexus/run.cjs detect_changes --scope compare --base-ref main
git diff --check
git status --short
git diff -- configs/deepseek_teacher.yaml tests/test_config.py
```

Expected: the cumulative implementation affects prompt construction and its known Teacher flows plus the active config/version assertion. No schema, training, export, platform-filtering, or data-cleaning files appear.

- [ ] **Step 8: Commit the cache-isolation change**

```bash
git add configs/deepseek_teacher.yaml tests/test_config.py
git commit -m "chore: bump Teacher protocol for Others prompts"
```

Expected: the commit contains exactly the two listed files.

- [ ] **Step 9: Verify the two-commit implementation history**

Run:

```bash
git log -2 --oneline
git diff --check HEAD~2 HEAD
git diff --name-only HEAD~2 HEAD
```

Expected file list:

```text
configs/deepseek_teacher.yaml
src/ibd/prompting.py
tests/test_config.py
tests/test_teacher.py
```

No generated data or cache directory is committed. Future Teacher generation with `configs/deepseek_teacher.yaml` uses the new prompt boundary and cannot reuse v4 cached calls.
