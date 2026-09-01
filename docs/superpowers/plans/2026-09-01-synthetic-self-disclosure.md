# Synthetic Self-disclosure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow brief, generic, low-risk synthetic supporter experience under the ESConv `Self-disclosure` strategy without weakening factual protections for the seeker or giving the strategy selection priority.

**Architecture:** Update the one shared strategy catalog and the Candidate-specific instructions, then add a narrow Final Selector exception so compliant synthetic experience is not rejected solely as unverifiable. Bump both Teacher protocols to isolate caches, synchronize the approved prompt-design documentation, and lock the behavior with prompt and configuration tests.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, YAML configuration, Markdown specifications.

## Global Constraints

- Work only on the current local branch `codex/implementation`.
- Preserve the uncommitted deletion of `AGENTS.md` and `CLAUDE.md`.
- Preserve the uncommitted `multi_view_state_analyzer.temperature: 0.3` change in `configs/deepseek_teacher_phase_balanced_v4.yaml` without including it in the feature commit.
- Keep all eight ESConv strategy names, including `Self-disclosure`.
- Synthetic experience must be brief, generic, low-risk, minimally detailed, and secondary to the seeker.
- Do not allow invented credentials, diagnosis or treatment history, self-harm, suicide, abuse, crime, severe trauma, protected identity, specific relationship or employment history, or externally verifiable biography.
- Do not weaken anti-fabrication rules for facts about the seeker, dialogue, other people, events, or external world.
- Do not modify `src/ibd/quality_judge.py` or add Self-disclosure rules to the common Student system prompt.
- Use protocol version `qwen25-socialsim-seven-state-policy-neutral-v2-synthetic-self-disclosure` in both Teacher configs.

---

### Task 1: Shared Strategy, Candidate, and Selector Semantics

**Files:**
- Modify: `src/ibd/prompting.py`
- Modify: `tests/test_teacher.py`

**Interfaces:**
- Consumes: `ESCONV_STRATEGIES`, `_CANDIDATE_PROMPT`, and `_ROLE_PROMPTS["final_selector"]` in `ibd.prompting`.
- Produces: one synchronized synthetic Self-disclosure definition used by Planner and Candidate, plus a Final Selector rule that permits only compliant synthetic experience.

- [ ] **Step 1: Replace the old Self-disclosure prompt test with failing synthetic-experience assertions**

```python
def test_self_disclosure_allows_only_low_risk_synthetic_experience(history):
    prompt = build_messages(
        "candidate",
        history,
        Candidate,
        context={
            "state": VALID_STATE,
            "candidate_id": "1",
            "strategy_id": "S1",
            "strategy": "Self-disclosure",
        },
    )[0]["content"]
    assert "brief, generic, low-risk synthetic supporter experience" in prompt
    assert "I went through something similar" in prompt
    assert "professional qualifications" in prompt
    assert "self-harm, suicide, abuse, crime, or severe trauma" in prompt
    assert "Do not invent facts about the seeker" in prompt
    assert "Do not claim personal history" not in prompt


def test_selector_does_not_treat_compliant_synthetic_disclosure_as_unsupported(history):
    prompt = build_messages(
        "final_selector",
        history,
        FinalSelectionDecision,
        context={"state": VALID_STATE, "candidates": []},
    )[0]["content"]
    assert "permitted generic synthetic supporter experience" in prompt
    assert "solely because it is not factually verifiable" in prompt
    assert "does not receive preference" in prompt
```

- [ ] **Step 2: Run the prompt tests and verify the old non-autobiographical wording fails**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_teacher.py -q -k 'self_disclosure or synthetic_disclosure'`

Expected: FAIL because the active catalog and Candidate prompt still prohibit personal or lived experience, and the selector has no synthetic-experience exception.

- [ ] **Step 3: Update the shared strategy catalog and Candidate prompt**

Use this shared catalog definition:

```python
"Self-disclosure": (
    "Use a brief, generic, low-risk synthetic supporter experience that is "
    "analogous to the seeker's situation, then return the focus to the seeker. "
    "Keep details minimal and do not use the experience as evidence or authority."
),
```

Replace the Candidate Self-disclosure procedure with explicit permission for forms such as `I went through something similar` and `I've felt overwhelmed in a situation like that too`. Add the exact prohibited categories from the global constraints. Retain the general prohibition on invented facts about the seeker, dialogue, other people, events, and external world.

- [ ] **Step 4: Add the narrow Final Selector exception**

After the general unsupported-fact exclusion, state that a permitted generic synthetic supporter experience in an assigned Self-disclosure Candidate is not excluded solely because it is not factually verifiable. Exclude it when it crosses a prohibited category, is used as authority, recenters the exchange, conflicts with dialogue or STATE, or otherwise fails conditional fit. State that it does not receive preference during quality comparison.

- [ ] **Step 5: Run all Teacher tests**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_teacher.py -q`

Expected: PASS.

---

### Task 2: Protocol Isolation, Documentation, and Regression

**Files:**
- Modify: `configs/deepseek_teacher.yaml`
- Modify: `configs/deepseek_teacher_phase_balanced_v4.yaml`
- Modify: `docs/superpowers/specs/2026-09-01-policy-neutral-prompt-redesign-design.md`
- Modify: `docs/superpowers/plans/2026-09-01-policy-neutral-prompt-redesign.md`
- Modify: `README.md`
- Modify: `tests/test_teacher.py`
- Test: `tests/test_student_data.py`

**Interfaces:**
- Consumes: `AppConfig.protocol_version` and `STUDENT_SYSTEM_PROMPT`.
- Produces: a new Teacher cache namespace and documentation consistent with the runtime prompts.

- [ ] **Step 1: Change the protocol test to require version 2**

```python
def test_teacher_configs_use_synthetic_self_disclosure_protocol():
    expected = (
        "qwen25-socialsim-seven-state-policy-neutral-"
        "v2-synthetic-self-disclosure"
    )
    for path in TEACHER_CONFIGS:
        assert AppConfig.from_yaml(path).protocol_version == expected
```

- [ ] **Step 2: Run the protocol test and verify it fails on version 1**

Run: `PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_teacher.py -q -k protocol`

Expected: FAIL because both configs still use `qwen25-socialsim-seven-state-policy-neutral-v1`.

- [ ] **Step 3: Bump both Teacher protocols without staging the user's temperature change**

Set:

```yaml
protocol_version: qwen25-socialsim-seven-state-policy-neutral-v2-synthetic-self-disclosure
```

Use partial staging for `configs/deepseek_teacher_phase_balanced_v4.yaml` so `temperature: 0.3` remains an uncommitted user change.

- [ ] **Step 4: Synchronize existing design, plan, and README**

Replace non-autobiographical Self-disclosure requirements in the 2026-09-01 policy-neutral design and plan with the synthetic-experience boundary from the approved delta specification. Document that version-1 traces, interventions, and anchors must not be reused for version 2. Do not add the policy to `STUDENT_SYSTEM_PROMPT`.

- [ ] **Step 5: Run static contradiction scans**

Run:

```bash
rg -n 'present reaction, stance, or engagement|do not claim personal history|Self-disclosure does not claim lived experience|Self-disclosure cannot claim personal history' src README.md docs/superpowers/specs/2026-09-01-policy-neutral-prompt-redesign-design.md docs/superpowers/plans/2026-09-01-policy-neutral-prompt-redesign.md
```

Expected: no matches.

- [ ] **Step 6: Run regression verification**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m compileall -q src
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest -q --deselect=tests/test_config.py::test_default_config_loads_as_explicit_compact_teacher_protocol --deselect=tests/test_qwen.py::test_checked_in_standard_sft_control_config_matches_b_8e_5
```

Expected: all selected tests PASS. The two deselected tests are pre-existing HEAD inconsistencies unrelated to this feature.

- [ ] **Step 7: Commit the implementation**

Stage runtime prompts, synchronized documents, tests, both protocol-version hunks, and README. Do not stage `AGENTS.md`, `CLAUDE.md`, the phase-balanced temperature hunk, or `src/ibd/quality_judge.py`.

```bash
git commit -m "feat: allow synthetic self-disclosure examples"
```

