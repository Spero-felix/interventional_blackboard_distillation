# SocialSim Phase-Stratified Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace last-turn-only SSConv preparation with deterministic 10% early, 80% middle, and 10% late response-target sampling, preserve v1 readability, and document downstream policy-collapse evidence.

**Architecture:** `src/ibd/socialsim.py` owns dialogue parsing, phase binning, exact quota allocation, metadata validation, and artifact construction. `src/ibd/cli.py` exposes only the two tail fractions and passes them into the preparation API. Existing Teacher loading remains schema-driven and accepts both v1 and v2 prepared artifacts.

**Tech Stack:** Python 3.11, Pydantic v2, argparse, pytest.

## Global Constraints

- Default phase mix is early `0.1`, middle `0.8`, late `0.1` independently in each split.
- Every selected conversation contributes exactly one example and conversation IDs never cross splits.
- A history contains only turns strictly before its selected Supporter target.
- Source ordering must not affect seeded output.
- New artifacts use `socialsim-qwen-conversation-v2`; v1 artifacts remain readable.
- Existing `artifacts/full-2000/` files must not be overwritten.
- Teacher prompts, intervention acceptance, and Stage A/B/C losses are out of scope for this patch.

---

### Task 1: Phase-aware SocialSim preparation

**Files:**
- Modify: `tests/test_socialsim.py`
- Modify: `src/ibd/socialsim.py`

**Interfaces:**
- Produces: `DialoguePhase = Literal["early", "middle", "late"]`.
- Produces: `prepare_socialsim_examples(..., early_fraction: float = 0.1, late_fraction: float = 0.1) -> PreparedSocialSim`.
- Produces: optional legacy-compatible `SocialSimExample.conversation_phase`, `target_turn`, `target_rank`, and `eligible_target_count` fields; v2 validation requires all four.
- Consumes: existing `DialogueTurn`, `History`, split sizes, seed, and SocialSim raw mappings.

- [ ] **Step 1: Replace the two-turn test fixture with an auditable multi-target fixture**

Change `_dialogue` in `tests/test_socialsim.py` to emit 12 alternating Seeker/Supporter pairs. Use literal content patterns `seeker-{dialogue_id}-{rank}`, `support-{dialogue_id}-{rank}`, and `REASONING-{dialogue_id}-{rank}` so a selected `target_rank` can be checked without calling production helpers.

- [ ] **Step 2: Write a failing exact-distribution and leakage test**

Add a test using 100 conversations and split sizes `60/20/20`. Assert literal phase counts of `10/80/10` globally and `6/48/6`, `2/16/2`, `2/16/2` by split. For every example, assert:

```python
assert len(example.history.turns) == example.target_turn - 1
assert example.history.turns[-1].role == "seeker"
assert 1 <= example.target_rank <= example.eligible_target_count == 12
serialized = json.dumps(example.history.model_dump(mode="json"))
assert f"support-{example.conversation_id}-{example.target_rank}" not in serialized
assert f"REASONING-{example.conversation_id}-{example.target_rank}" not in serialized
```

Derive the expected rank ranges by hand for 12 targets: early `1..4`, middle `5..8`, late `9..12`.

- [ ] **Step 3: Run the distribution test and verify RED**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_socialsim.py::test_socialsim_phase_sampling_has_exact_distribution_without_future_leakage -q
```

Expected: FAIL because `SocialSimExample` has no phase or target metadata and all current histories use the last target.

- [ ] **Step 4: Write failing validation and legacy-contract tests**

Add parametrized cases for `early_fraction=-0.1`, `late_fraction=float("nan")`, and `early_fraction=0.6, late_fraction=0.4`; each must raise `ValueError` mentioning phase fractions. Add a three-pair valid fixture plus a two-pair short fixture and assert the short ID appears in `manifest["excluded"]`. Construct a literal v1 payload with missing phase fields and assert `PreparedSocialSim.model_validate` accepts it; construct an equivalent v2 payload and assert it is rejected for missing phase metadata.

- [ ] **Step 5: Run the new validation tests and verify RED**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_socialsim.py -q
```

Expected: FAIL on missing fraction arguments, missing v2 contract enforcement, and lack of short-conversation exclusion.

- [ ] **Step 6: Implement the minimal phase-aware data contract**

In `src/ibd/socialsim.py`:

- add `math`, `dataclass`, `Counter`, and phase aliases/constants;
- change `PreparedSocialSim.protocol_version` to `Literal["socialsim-qwen-conversation-v1", "socialsim-qwen-conversation-v2"]`, defaulting to v2;
- add the four optional fields to `SocialSimExample`, with positive integer constraints;
- add a model validator that requires either all phase fields or none, verifies `target_rank <= eligible_target_count`, and verifies the recorded phase matches the rank bin;
- extend `PreparedSocialSim.validate_conversation_splits` to require metadata for v2 and validate manifest phase counts against actual examples.

Define focused private helpers:

```python
def _parse_dialogue(raw: Mapping[str, Any]) -> tuple[list[DialogueTurn], tuple[int, ...]]: ...
def _phase_for_rank(target_rank: int, target_count: int) -> DialoguePhase: ...
def _validate_phase_fractions(early_fraction: float, late_fraction: float) -> dict[DialoguePhase, float]: ...
def _phase_counts(size: int, fractions: Mapping[DialoguePhase, float]) -> dict[DialoguePhase, int]: ...
```

`_parse_dialogue` returns normalized turns plus zero-based Supporter target indices and rejects fewer than three targets. `_phase_for_rank` uses `min(2, ((target_rank - 1) * 3) // target_count)` so 12 ranks map to `1..4`, `5..8`, `9..12`. `_phase_counts` floors all three exact products and awards remaining rows by descending fractional remainder with `_PHASE_ORDER` as the deterministic tie-break.

- [ ] **Step 7: Implement deterministic split-local selection**

Parse all conversations into `(conversation_id, turns, target_indices)`, sort by ID, shuffle once with `random.Random(seed)`, and slice the selected conversations by split. For each split, create the exact number of phase labels from `_phase_counts`, shuffle them with `random.Random(seed + (split_index + 1) * 1_000_003)`, and for each assigned conversation uniformly choose one target index whose one-based rank maps to that phase. Export `History(turns=turns[:target_index])` and all four metadata fields.

Return a v2 manifest containing literal keys `phase_fractions`, `phase_counts`, and `split_phase_counts`, in addition to existing counts, IDs, exclusions, and profile-usage fields.

- [ ] **Step 8: Run the SocialSim tests and verify GREEN**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_socialsim.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Refactor without changing behavior and re-run tests**

Keep phase allocation, target selection, and model validation in separate helpers. Run the Task 1 test command again and require exit code 0.

- [ ] **Step 10: Commit Task 1**

```bash
git add src/ibd/socialsim.py tests/test_socialsim.py
git commit -m "fix: stratify SocialSim dialogue phases"
```

---

### Task 2: CLI v2 integration and backward compatibility

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/ibd/cli.py`

**Interfaces:**
- Consumes: Task 1's `prepare_socialsim_examples` fraction arguments and v1/v2 `PreparedSocialSim` validation.
- Produces: `prepare-socialsim --early-fraction FLOAT --late-fraction FLOAT`.

- [ ] **Step 1: Expand the CLI fixture and write a failing v2 artifact test**

Change `_socialsim_dialogue` to emit six alternating pairs. Update `test_prepare_socialsim_cli_writes_reproducible_artifact` to use 20 rows and split sizes `10/5/5`, pass `--early-fraction 0.1 --late-fraction 0.1`, and assert:

```python
assert payload["protocol_version"] == "socialsim-qwen-conversation-v2"
assert payload["manifest"]["phase_counts"] == {"early": 2, "middle": 16, "late": 2}
assert all(row["conversation_phase"] for rows in payload["splits"].values() for row in rows)
```

Retain the assertion that profile secrets are absent.

- [ ] **Step 2: Run the CLI artifact test and verify RED**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_cli.py::test_prepare_socialsim_cli_writes_reproducible_artifact -q
```

Expected: argparse rejects the two new flags.

- [ ] **Step 3: Implement the CLI arguments and v2 loader**

Add the two float arguments with defaults `0.1` and pass them to `load_socialsim_files`. Change `_load_records` to recognize both supported SocialSim protocol strings before validating through `PreparedSocialSim`; do not duplicate schema validation in the CLI.

- [ ] **Step 4: Verify v1 Teacher input and v2 CLI output**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_cli.py::test_run_teacher_accepts_prepared_socialsim_artifact tests/test_cli.py::test_prepare_socialsim_cli_writes_reproducible_artifact -q
```

Expected: both tests pass, proving v1 read compatibility and v2 write behavior.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/ibd/cli.py tests/test_cli.py
git commit -m "feat: expose SocialSim phase fractions"
```

---

### Task 3: Production-data verification and user documentation

**Files:**
- Modify: `README.md`
- Do not overwrite: `artifacts/full-2000/prepared.json`

**Interfaces:**
- Consumes: Task 2's CLI and the raw 3,229-conversation SSConv/profile files.
- Produces: a temporary v2 artifact under `/tmp` used only for verification.

- [ ] **Step 1: Update the README command and expected manifest**

Document `--limit 2000 --train-size 1500 --dev-size 200 --holdout-size 300 --early-fraction 0.1 --late-fraction 0.1`. State that the expected split-local counts are train `150/1200/150`, dev `20/160/20`, and diagnostic holdout `30/240/30`, and that old Teacher traces/interventions cannot be paired with the new contexts.

- [ ] **Step 2: Generate a fresh temporary production-sized artifact**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m ibd.cli prepare-socialsim \
  --dialogues /home/wangnianxiang/supervisor/data/raw/SocialSim_SSConv_ESC_dataset_full_3229.json \
  --profiles /home/wangnianxiang/supervisor/data/raw/SocialSim_UserProfile_full_3229.json \
  --output /tmp/ibd-socialsim-phase-balanced-2000.json \
  --seed 42 --limit 2000 --train-size 1500 --dev-size 200 --holdout-size 300 \
  --early-fraction 0.1 --late-fraction 0.1
```

Expected: exit code 0 and `wrote 2000 SocialSim conversations`.

- [ ] **Step 3: Independently verify counts, ranks, and leakage**

Run a read-only Python check that loads the raw dialogue file and temporary artifact, asserts the literal global and split counts, asserts `len(history.turns) == target_turn - 1`, and asserts the raw target Supporter text is absent from serialized history for all 2,000 examples. Print the min/median/max target progress to confirm middle contexts are present.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_socialsim.py tests/test_cli.py -q
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest -q
```

Expected: both commands exit 0 with no failures.

- [ ] **Step 5: Commit Task 3**

```bash
git add README.md
git commit -m "docs: document phase-balanced SocialSim build"
```

---

### Task 4: Downstream policy-collapse audit

**Files:**
- Create: `docs/superpowers/reports/2026-08-30-policy-collapse-audit.md`

**Interfaces:**
- Consumes: `artifacts/full-2000/teacher-traces.jsonl`, `artifacts/full-2000/interventions.jsonl`, Teacher prompting/selection code, intervention filtering code, and Stage A/B/C training code/config.
- Produces: an evidence report; no runtime behavior changes.

- [ ] **Step 1: Measure each artifact boundary**

Use a read-only Python command to calculate split counts, planned-strategy frequency, final selected-strategy frequency, candidate-position frequency, response duplication/length, intervention retention overall and by selected strategy, PLAN/STATE balance, and train/dev coverage. Record denominators as well as percentages.

- [ ] **Step 2: Trace measured bias to code mechanisms**

Read and cite exact source lines for planner prompts, final-selector criteria, intervention exclusion gates, Stage A/B replay exposure, Stage C function sampling, loss weights, and checkpoint-selection metrics. Label each item as `observed`, `likely amplifier`, or `monitoring gap`; do not describe an unmeasured mechanism as a root cause.

- [ ] **Step 3: Write prioritized recommendations**

The report must separate immediate gates for the regenerated dataset from later experimental changes. Immediate gates include maximum selected-strategy share, per-phase strategy tables, intervention retention by strategy/phase, and generation-diversity checks. Any recommendation to rebalance Teacher selection or loss weights must be conditional on the new phase-balanced traces still showing collapse.

- [ ] **Step 4: Verify report numbers against fresh commands**

Re-run the audit command, compare every reported number, and run `git diff --check` on the report.

- [ ] **Step 5: Commit Task 4**

```bash
git add -f docs/superpowers/reports/2026-08-30-policy-collapse-audit.md
git commit -m "docs: audit downstream policy collapse risks"
```

---

### Task 5: Final verification

**Files:**
- Verify all files changed by Tasks 1-4.

**Interfaces:**
- Consumes: completed implementation and report.
- Produces: fresh completion evidence.

- [ ] **Step 1: Inspect scoped changes**

Run `git status --short`, `git diff --check`, and scoped diffs for `src/ibd/socialsim.py`, `src/ibd/cli.py`, `tests/test_socialsim.py`, `tests/test_cli.py`, `README.md`, and the report. Confirm unrelated pre-existing modifications were not staged or rewritten.

- [ ] **Step 2: Run the complete test suite fresh**

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest -q
```

Expected: exit code 0 and zero failed tests.

- [ ] **Step 3: Re-run the 2,000-row artifact audit fresh**

Repeat Task 3 Step 3 against `/tmp/ibd-socialsim-phase-balanced-2000.json`. Expected global counts are exactly early 200, middle 1,600, late 200, with no target-response leakage.

- [ ] **Step 4: Summarize implementation and residual risks**

Report changed files, exact phase evidence, tests executed, current downstream collapse evidence, and the fact that existing Teacher traces/interventions must be regenerated from the new prepared artifact.
