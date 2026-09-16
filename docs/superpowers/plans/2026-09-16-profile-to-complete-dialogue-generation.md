# Profile-to-Complete-Dialogue Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a batch workflow in which one external input profile drives one complete seeker–supporter conversation, while preserving the existing full Teacher pipeline and keeping non-closing hard-limit results out of completed training data.

**Architecture:** A profile adapter exposes only `profile_id` and seeker-rendered private text. `SeekerSimulator` generates the next public seeker utterance; `TeacherSession.observe()` updates Context and computes the existing `MultiViewStateAnalysis`; an extra `DialogueManager` call selects a soft `DialogueMode`; `TeacherSession.respond()` runs PLAN, candidates, and final selection. `ConversationGenerator` owns alternation, deterministic seeds, termination, checkpoints, and round-bound audit records. Raw profile data never enters manager or supporter prompts.

**Tech Stack:** Python 3.11, Pydantic v2 strict models, the existing OpenAI-compatible `LLMBackend`, JSON/JSONL persistence, PyYAML configuration, pytest.

## Global Constraints

- Preserve `TeacherRunner.run()`, current `TeacherTrace`, `run-teacher`, exports, and Stage A/B inputs.
- Do not define the final profile schema. The first adapter accepts an opaque mapping with one required non-empty `ID` and renders the complete mapping only for the seeker.
- Do not add `SeekerPrivateState`, `ProfileConsistencyChecker`, `DialogueGoal`, `should_close`, or a reduced-cost supporter path.
- Raw profile content may appear only in `seeker_simulator` messages. It must not appear in Context, STATE, manager, planner, candidate, selector, public turns, checkpoints, or final traces unless the seeker model itself says it publicly.
- Reuse `MultiViewStateAnalysis`, `StateBlackboard`, `StateEvidence`, and `TeacherTrace`; conversation code must not create a second STATE representation.
- Treat `DialogueMode` as a weak orientation. No transition table, monotonic phase rule, or fixed mode-to-strategy mapping is permitted.
- Only `decision.mode == "closing"` creates `status="completed"`. A non-closing hard maximum creates `status="truncated"`, even after generating the final supporter turn.
- Persist one `ConversationRoundAudit` per complete seeker–supporter round; do not use parallel decision and trace arrays.
- A `TeacherSession` commits `context_after` only after PLAN/candidates/selection succeed, so a failed partial round cannot advance session memory.
- Use SHA-256-derived, signed-31-bit seeds; never use Python's process-randomized `hash()`.
- All implementation changes follow red-green-refactor: add a focused failing test, run it and confirm the intended failure, implement the minimum behavior, then run the focused and regression tests.

## File and Responsibility Map

| File | Responsibility after this change |
|---|---|
| `src/ibd/schemas.py` | Existing Teacher schemas plus `TurnObservation`, `TurnResponse`, and `TeacherTrace.from_parts()` |
| `src/ibd/conversation_schemas.py` | Modes, generation limits, turns, decisions, round audits, termination, checkpoints, traces, and failures |
| `src/ibd/seeding.py` | Stable per-conversation/per-round/per-role seed derivation |
| `src/ibd/backend.py` | Shared cache primitive, structured-call error records, existing structured caller |
| `src/ibd/seeker.py` | Opaque profile adapter protocols/implementation, non-JSON `TextCaller`, seeker prompt and simulator |
| `src/ibd/dialogue_manager.py` | Structured manager call and invalid-schema fallback to the previous mode |
| `src/ibd/teacher.py` | Observe/respond split, optional seed derivation, session transaction boundary, legacy wrapper |
| `src/ibd/prompting.py` | Dialogue Manager prompt and Planner's mode weak-prior clause |
| `src/ibd/conversation.py` | Full round loop, stage-aware errors, dynamic termination, trace flattening |
| `src/ibd/storage.py` | Atomic JSON checkpoints and existing JSONL helpers |
| `src/ibd/cli.py` | Batch generation and flattening commands |
| `tests/test_*.py` | Focused schema, seed, backend, seeker, manager, orchestration, persistence, privacy, and CLI tests |
| `README.md` | User-facing architecture, command, status, resume, and output documentation |

---

### Task 1: Define conversation contracts and enforce invariants

**Files:**

- Create: `src/ibd/conversation_schemas.py`
- Modify: `src/ibd/schemas.py`
- Create: `tests/test_conversation_schemas.py`
- Modify: `tests/test_schemas.py`

**Interfaces:**

```python
DialogueMode = Literal["opening", "exploration", "comforting", "action", "closing"]
ConversationStatus = Literal["completed", "truncated"]
TerminationReason = Literal[
    "dialogue_manager_closing",
    "hard_max_without_closing",
]

class ConversationGenerationConfig(StrictModel):
    min_rounds: int = Field(default=6, ge=1)
    soft_max_rounds: int = Field(default=16, ge=1)
    hard_max_rounds: int = Field(default=20, ge=1)
    split: Literal["train", "dev", "diagnostic_holdout"] = "train"

class DialogueManagementDecision(StrictModel):
    mode: DialogueMode
    transition_reason: str = Field(min_length=1)

class DialogueDecisionRecord(StrictModel):
    round_index: int = Field(ge=1)
    mode_before: DialogueMode
    decision: DialogueManagementDecision
    call_records: list[CallRecord]

class ConversationRoundAudit(StrictModel):
    round_index: int = Field(ge=1)
    seeker_turn_index: int = Field(ge=1)
    supporter_turn_index: int = Field(ge=1)
    dialogue_decision: DialogueDecisionRecord
    teacher_trace: TeacherTrace

class ConversationTurn(StrictModel):
    turn_index: int = Field(ge=1)
    round_index: int = Field(ge=1)
    role: Literal["seeker", "supporter"]
    content: str = Field(min_length=1)

class ConversationTermination(StrictModel):
    reason: TerminationReason
    round_count: int = Field(ge=1)
    final_mode: DialogueMode

class ConversationTrace(StrictModel):
    protocol_version: str
    conversation_id: str
    profile_id: str
    seed: int
    split: Literal["train", "dev", "diagnostic_holdout"]
    status: ConversationStatus
    turns: list[ConversationTurn]
    rounds: list[ConversationRoundAudit]
    termination: ConversationTermination
    seeker_call_records: list[CallRecord]

class ConversationCheckpoint(StrictModel):
    protocol_version: str
    conversation_id: str
    profile_id: str
    seed: int
    generation_config: ConversationGenerationConfig
    turns: list[ConversationTurn]
    rounds: list[ConversationRoundAudit]
    last_context: UserContext
    previous_mode: DialogueMode
    next_round_index: int = Field(ge=1)
    seeker_call_records: list[CallRecord]

class ConversationFailure(StrictModel):
    conversation_id: str
    profile_id: str
    seed: int
    failed_stage: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    error: str = Field(min_length=1)
    completed_rounds: int = Field(ge=0)
    checkpoint_path: str | None = None
```

`ConversationCheckpoint` deliberately stores only public/generated state and the last committed `UserContext`; it never stores the rendered profile.

`src/ibd/schemas.py` adds:

```python
class TurnObservation(StrictModel):
    example_id: str
    history: History
    context_before: UserContext
    context_patch: ContextPatch
    context_after: UserContext
    context_merge_errors: list[str]
    state_analysis: MultiViewStateAnalysis
    call_records: list[CallRecord]

class TurnResponse(StrictModel):
    plan: StrategyPlanSet
    candidates: Annotated[list[Candidate], Field(min_length=1, max_length=3)]
    final_selection: FinalSelection
    call_records: list[CallRecord]
```

- [ ] Write tests that reject `min_rounds > soft_max_rounds`, `soft_max_rounds > hard_max_rounds`, non-alternating turns, missing round audits, non-contiguous indexes, a teacher history that differs from the public prefix ending at that round's seeker turn, and a completed trace whose final mode is not `closing`.

- [ ] Add positive fixtures for one completed trace and one hard-max truncated trace. Assert that both begin with seeker, strictly alternate, end with supporter, contain exactly two turns per round, and bind each supporter turn to one audit.

Representative failing assertion:

```python
with pytest.raises(ValueError, match="completed conversation must end in closing"):
    ConversationTrace.model_validate({**payload, "status": "completed", ...})
```

- [ ] Run the new test before implementation:

```bash
.venv/bin/pytest -q tests/test_conversation_schemas.py tests/test_schemas.py
```

Expected: collection fails because `ibd.conversation_schemas`, `TurnObservation`, and `TurnResponse` do not exist.

- [ ] Implement strict models and model validators. Required cross-field checks:

  - generation limits are ordered;
  - turn and round indexes start at one and are contiguous;
  - role alternation is exact;
  - `len(turns) == 2 * len(rounds)`;
  - audit turn indexes identify that round's seeker and supporter;
  - `dialogue_decision.round_index == audit.round_index`;
  - `teacher_trace.history` equals all public turns through the current seeker turn, converted to `History`;
  - `teacher_trace.final_response` equals the bound supporter content;
  - completed implies `dialogue_manager_closing` and final mode `closing`;
  - truncated implies `hard_max_without_closing` and final mode is not `closing`;
  - checkpoint state always ends after a complete supporter round, or is empty before round one.

- [ ] Add `TeacherTrace.from_parts(observation, response, *, split)` and test that it concatenates observation/response call records and still passes the existing provenance validator.

- [ ] Run focused tests:

```bash
.venv/bin/pytest -q tests/test_conversation_schemas.py tests/test_schemas.py
```

Expected: pass.

- [ ] Commit:

```bash
git add src/ibd/conversation_schemas.py src/ibd/schemas.py tests/test_conversation_schemas.py tests/test_schemas.py
git commit -m "feat: define complete conversation trace contracts"
```

---

### Task 2: Add stable call-seed derivation

**Files:**

- Create: `src/ibd/seeding.py`
- Create: `tests/test_seeding.py`

**Interface:**

```python
@dataclass(frozen=True)
class SeedDeriver:
    protocol_version: str
    conversation_id: str
    base_seed: int
    round_index: int

    def for_call(self, role: str, cache_variant: str | None = None) -> int: ...
```

- [ ] Write tests asserting identical inputs give identical seeds, changing any identity component changes the seed, candidate variants `S1`/`S2` differ, values are in `0 <= seed < 2**31`, and replacing `builtins.hash` with a function that raises does not affect derivation.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_seeding.py
```

Expected: import failure for `ibd.seeding`.

- [ ] Implement canonical UTF-8 serialization of the five components separated by a fixed delimiter, SHA-256 hashing, and conversion from the first eight digest bytes modulo `2**31`. Reject an empty role and `round_index < 1`.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_seeding.py
```

Expected: pass.

- [ ] Commit:

```bash
git add src/ibd/seeding.py tests/test_seeding.py
git commit -m "feat: derive stable per-round model seeds"
```

---

### Task 3: Extract reusable call caching and retain failed structured records

**Files:**

- Modify: `src/ibd/backend.py`
- Modify: `tests/test_backend.py`

**Interfaces:**

```python
class StructuredCallError(ValueError):
    role: str
    records: tuple[CallRecord, ...]

class CallCache:
    def key(self, example_id: str, role: str, cache_variant: str | None = None) -> str: ...
    def read(self, key: str) -> dict[str, Any] | None: ...
    def write(self, key: str, payload: dict[str, Any], *, failure: bool = False) -> None: ...
```

- [ ] Extend backend tests to assert:

  - the existing no-variant cache digest remains byte-for-byte unchanged;
  - success records still replay with `cached=True`;
  - failure cache remains under `failures/`;
  - exhausted invalid JSON raises a `StructuredCallError` that is also a `ValueError` and exposes both attempt records.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_backend.py
```

Expected: failure because `StructuredCallError` and `CallCache` are absent.

- [ ] Move namespace/key/path/atomic-write/raw-read behavior into `CallCache`. Keep `StructuredCaller._cache_key()` as a delegating compatibility method because the current tests and callers use it.

- [ ] Change only the exhausted schema-validation branch to raise `StructuredCallError`; do not wrap `EmptyContentError` or provider exceptions. Preserve existing failure JSON payload and all successful `CallRecord` fields.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_backend.py tests/test_teacher.py
```

Expected: pass, including current candidate fallback and context-updater behavior.

- [ ] Commit:

```bash
git add src/ibd/backend.py tests/test_backend.py
git commit -m "refactor: share model call cache and expose retry records"
```

---

### Task 4: Implement the opaque profile boundary and Seeker Simulator

**Files:**

- Create: `src/ibd/seeker.py`
- Create: `tests/test_seeker.py`

**Interfaces:**

```python
class ValidatedProfile(Protocol):
    @property
    def profile_id(self) -> str: ...
    def render_for_seeker(self) -> str: ...

class SeekerProfileAdapter(Protocol):
    def validate(self, raw_profile: Mapping[str, Any]) -> ValidatedProfile: ...

class MappingProfileAdapter:
    def validate(self, raw_profile: Mapping[str, Any]) -> ValidatedProfile: ...

class TextCaller:
    def call(
        self,
        role: str,
        messages: list[dict[str, str]],
        *,
        seed: int,
        example_id: str,
        cache_variant: str | None = None,
    ) -> tuple[str, CallRecord]: ...

@dataclass(frozen=True)
class SeekerGeneration:
    utterance: str
    call_record: CallRecord

class SeekerSimulator:
    def generate(
        self,
        profile: ValidatedProfile,
        history: Sequence[ConversationTurn],
        mode: DialogueMode,
        *,
        round_index: int,
        seed: int,
        conversation_id: str,
    ) -> SeekerGeneration: ...
```

- [ ] Write adapter tests for integer/string IDs, blank or missing IDs, immutable copied payloads, deterministic seeker rendering, and absence of profile payload fields on the public interface.

- [ ] Write `TextCaller` tests proving `json_mode=False`, whitespace trimming, rejection/retry of empty text, retry of provider exceptions, no retry for successful non-JSON prose, persisted cache replay with `cached=True`, and a `CallRecord.parsed == {"text": stripped_text}`.

- [ ] Write seeker prompt tests asserting it contains the complete private rendering, public history, mode, and round index, but contains none of `min_rounds`, `soft_max_rounds`, `hard_max_rounds`, Context, STATE, PLAN, candidate, selector, or transition reason fields. Assert the first round accepts empty public history.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_seeker.py
```

Expected: import failure for `ibd.seeker`.

- [ ] Implement `MappingProfileAdapter` as the deliberately minimal first adapter: normalize `ID` to a stripped string, deep-copy the mapping, and render stable `json.dumps(..., ensure_ascii=False, sort_keys=True)`. Do not interpret any other profile field.

- [ ] Implement the approved seeker system contract verbatim in substance: seeker-only role, private background, gradual disclosure, consistency, preserved uncertainty, permission to reject support, no automatic satisfaction, natural utterance-level closure, and output-only-next-utterance.

- [ ] Implement `TextCaller` using `CallCache`, the role's `ModelConfig`, and `config.schema_retries + 1` attempts. Catch normal `Exception` values raised by the provider boundary but never catch `KeyboardInterrupt`/`SystemExit`; after exhaustion, raise the last provider/empty-output error. Record only a successful textual completion because provider exceptions have no response body.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_seeker.py tests/test_backend.py
```

Expected: pass.

- [ ] Commit:

```bash
git add src/ibd/seeker.py tests/test_seeker.py
git commit -m "feat: add private-profile seeker simulation"
```

---

### Task 5: Split Teacher observation from response without changing legacy behavior

**Files:**

- Modify: `src/ibd/teacher.py`
- Modify: `tests/test_teacher.py`

**Interfaces:**

```python
class TeacherRunner:
    def observe(
        self,
        example_id: str,
        history: History,
        *,
        context_before: UserContext | None = None,
        seed_deriver: SeedDeriver | None = None,
    ) -> TurnObservation: ...

    def respond(
        self,
        observation: TurnObservation,
        *,
        dialogue_mode: DialogueMode | None = None,
        seed_deriver: SeedDeriver | None = None,
    ) -> TurnResponse: ...

class TeacherSession:
    def observe(...) -> TurnObservation: ...
    def respond(
        self,
        observation: TurnObservation,
        *,
        split: Literal["train", "dev", "test", "diagnostic_holdout"] = "train",
        dialogue_mode: DialogueMode | None = None,
        seed_deriver: SeedDeriver | None = None,
    ) -> TeacherTrace: ...
```

- [ ] Add a test comparing `TeacherRunner.run()` with `observe()` plus `respond()` plus `TeacherTrace.from_parts()` for identical fields and call-role order.

- [ ] Add a test that `observe()` performs only `context_updater` and `multi_view_state_analyzer`, and `respond()` performs only planner, candidates, and selector.

- [ ] Add a transactional session test: make planner fail after a successful observation and assert `session.context` remains the old value; then run a successful response and assert it commits `observation.context_after`.

- [ ] Add seed tests showing legacy `run()` still passes `None`, while a supplied `SeedDeriver` gives stable distinct seeds to Context, analyzer, planner, each candidate variant, and selector.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_teacher.py
```

Expected: failures because `observe()` and `respond()` do not exist.

- [ ] Move the current Context update/reduce/STATE code unchanged into `observe()`. Move `_run_from_state()` orchestration into `respond()`. Thread optional derived seeds through `_call()`, using the role and candidate `cache_variant`.

- [ ] Rebuild `run()` strictly as `observe -> respond -> TeacherTrace.from_parts`. Rebuild `TeacherSession.run()` strictly as its transactional `observe -> respond` wrapper. `TeacherSession.observe()` must not mutate context; `TeacherSession.respond()` validates that the observation began at the current context and commits only after a full trace exists.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_teacher.py tests/test_export.py tests/test_cli.py
```

Expected: pass with `NORMAL_ROLES` unchanged.

- [ ] Commit:

```bash
git add src/ibd/teacher.py tests/test_teacher.py
git commit -m "refactor: split teacher observation and response"
```

---

### Task 6: Add LLM-based Dialogue Manager and mode-aware Planner input

**Files:**

- Create: `src/ibd/dialogue_manager.py`
- Modify: `src/ibd/prompting.py`
- Create: `tests/test_dialogue_manager.py`
- Modify: `tests/test_teacher.py`

**Interface:**

```python
class DialogueManager:
    def decide(
        self,
        observation: TurnObservation,
        *,
        previous_mode: DialogueMode,
        recent_selected_strategies: Sequence[StrategyName],
        round_index: int,
        generation_config: ConversationGenerationConfig,
        seed: int,
        conversation_id: str,
    ) -> DialogueDecisionRecord: ...
```

- [ ] Write manager payload tests asserting exact availability of public `history`, `user_context`, the complete existing `state_analysis`, previous mode, recent selected strategies, and all three round limits. Assert no profile, `DialogueGoal`, `should_close`, or fixed transition map appears.

- [ ] Parameterize decisions showing the manager can stay, advance, regress, and skip modes without code-side rejection.

- [ ] Test malformed JSON followed by valid JSON yields two call records. Test exhausted malformed JSON catches `StructuredCallError`, returns `previous_mode`, retains the failed call records, and writes an audit-only transition reason beginning `fallback_to_previous_mode_after_schema_failure:`.

- [ ] Test Planner messages receive only `dialogue_mode`, not `transition_reason`, and contain the weak-prior clause. Test candidate and selector messages do not receive either field.

- [ ] Update the existing exact `PROMPT_ROLES` assertion to include `dialogue_manager`; keep `NORMAL_ROLES` unchanged.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_dialogue_manager.py tests/test_teacher.py
```

Expected: failures because the manager role/class and Planner mode input do not exist.

- [ ] Add the manager prompt with the five mode definitions and explicit rules: modes are not a mandatory sequence; latest seeker evidence is primary; length alone cannot force action; brief acknowledgment cannot force closing; repetitive exploration should stop. Its structured output is exactly `DialogueManagementDecision`.

- [ ] Pass `dialogue_mode` only into the Planner context when non-`None`. Append the approved weak-prior policy to the Planner system prompt. Do not add mode to `_CONTEXT_CONSUMING_ROLES` or `_STATE_CONSUMING_ROLES`; Planner already belongs to both.

- [ ] Use cache identity `example_id=conversation_id` and `cache_variant=f"round-{round_index}"` for the manager call. Provider/transport errors propagate and fail the conversation; only exhausted schema validation falls back to the prior mode.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_dialogue_manager.py tests/test_teacher.py tests/test_backend.py
```

Expected: pass.

- [ ] Commit:

```bash
git add src/ibd/dialogue_manager.py src/ibd/prompting.py tests/test_dialogue_manager.py tests/test_teacher.py
git commit -m "feat: choose soft dialogue modes with an llm manager"
```

---

### Task 7: Implement the complete conversation round loop

**Files:**

- Create: `src/ibd/conversation.py`
- Create: `tests/test_conversation.py`

**Interfaces:**

```python
class ConversationStageError(RuntimeError):
    stage: Literal[
        "seeker_simulator",
        "teacher_observe",
        "dialogue_manager",
        "teacher_respond",
    ]
    completed_rounds: int

class ConversationGenerator:
    def generate(
        self,
        profile: ValidatedProfile,
        *,
        base_seed: int,
        checkpoint: ConversationCheckpoint | None = None,
        on_round_completed: Callable[[ConversationCheckpoint], None] | None = None,
    ) -> ConversationTrace: ...

def flatten_teacher_traces(trace: ConversationTrace) -> list[TeacherTrace]: ...
```

- [ ] Build a deterministic scripted integration backend and write a two-round test with manager modes `exploration`, then `closing`. Assert exact role order per round:

```text
seeker_simulator
context_updater
multi_view_state_analyzer
dialogue_manager
planner
candidate x N
final_selector
```

- [ ] Assert round one seeker uses `opening`, round one supporter uses manager-selected `exploration`, round two seeker uses `exploration`, and round two supporter uses `closing`. Assert transition reasons never reach seeker or supporter prompts.

- [ ] Add hard-max boundary tests:

  - non-closing at hard max still generates that round's supporter reply, then returns `status="truncated"` and `hard_max_without_closing`;
  - closing exactly at hard max returns `status="completed"`;
  - `min_rounds` and `soft_max_rounds` are manager inputs only and never override a decision;
  - no path mutates a non-closing decision to closing.

- [ ] Add audit tests for history prefixes, turn indexes, round indexes, selected-strategy history, context carry-forward, seeker call-record cardinality, and `flatten_teacher_traces(trace) == [round.teacher_trace ...]`.

- [ ] Add stage-failure parameterization. For each model boundary, raise a known exception and assert `ConversationStageError.stage`, original `__cause__`, and `completed_rounds`; assert no partial seeker-only round reaches `on_round_completed`.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_conversation.py
```

Expected: import failure for `ibd.conversation`.

- [ ] Implement the loop with `previous_mode="opening"` on a fresh conversation. For each round:

  1. derive round-scoped seeds;
  2. generate and append one seeker turn;
  3. construct `History` and call `TeacherSession.observe()`;
  4. call the manager with at most the three most recent selected strategies;
  5. call `TeacherSession.respond()` with only `decision.mode`;
  6. append the exact selected supporter response;
  7. bind decision and trace in `ConversationRoundAudit`;
  8. create and emit a complete-round checkpoint;
  9. classify closing first, hard max second, otherwise carry `decision.mode` forward.

- [ ] Derive `conversation_id` deterministically as `f"{profile.profile_id}-seed-{base_seed}"`; reject profile IDs containing control characters. Do not place rendered profile data in any generated model or exception object.

- [ ] On resume, validate checkpoint identity, protocol, seed, and generation config; initialize `TeacherSession` from `last_context`; use stored turns/rounds/call records/mode/index; never replay completed rounds.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_conversation.py tests/test_conversation_schemas.py tests/test_teacher.py
```

Expected: pass.

- [ ] Commit:

```bash
git add src/ibd/conversation.py tests/test_conversation.py
git commit -m "feat: generate complete seeker supporter conversations"
```

---

### Task 8: Add atomic checkpoints and resume-safe artifact routing

**Files:**

- Modify: `src/ibd/storage.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_conversation.py`

**Interfaces:**

```python
def write_json_atomic(path: str | Path, record: Any) -> None: ...
def read_json(path: str | Path) -> dict[str, Any]: ...
def conversation_checkpoint_path(
    checkpoint_dir: str | Path,
    conversation_id: str,
) -> Path: ...
```

- [ ] Write storage tests proving atomic JSON writes create parent directories, round-trip UTF-8/Pydantic values, leave no temporary file, and replace an existing checkpoint completely. Assert checkpoint filenames are a SHA-256 digest plus `.json`, so profile IDs cannot escape the checkpoint directory.

- [ ] Add an interrupted-generation test: stop after round N from the checkpoint callback, load the persisted checkpoint, resume with a fresh generator/backend, and assert the final trace equals an uninterrupted run while no first-N-round model calls repeat.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_storage.py tests/test_conversation.py
```

Expected: failures because atomic checkpoint helpers are absent.

- [ ] Implement atomic JSON with `mkstemp` in the target directory, flush, file `fsync`, `os.replace`, and directory `fsync`, matching the durability pattern already used for model-call cache. `read_json` must require one JSON object.

- [ ] Make the integration callback write `ConversationCheckpoint` after every complete round. Keep the checkpoint after completion/truncation as a reproducibility artifact; finalized-output ID checks, not checkpoint deletion, prevent duplicate generation.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_storage.py tests/test_conversation.py
```

Expected: pass.

- [ ] Commit:

```bash
git add src/ibd/storage.py tests/test_storage.py tests/test_conversation.py
git commit -m "feat: persist resumable conversation checkpoints"
```

---

### Task 9: Add batch CLI, separate outputs, failures, and flatten export

**Files:**

- Modify: `src/ibd/cli.py`
- Modify: `tests/test_cli.py`

**Commands:**

```bash
python -m ibd.cli generate-conversations \
  --config configs/deepseek_teacher.yaml \
  --profiles data/profiles.json \
  --output artifacts/conversations/conversations.jsonl \
  --truncated artifacts/conversations/truncated-conversations.jsonl \
  --failures artifacts/conversations/conversation-failures.jsonl \
  --checkpoint-dir artifacts/conversations/checkpoints \
  --seed 42 \
  --split train \
  --min-rounds 6 \
  --soft-max-rounds 16 \
  --hard-max-rounds 20 \
  --resume

python -m ibd.cli flatten-conversations \
  --input artifacts/conversations/conversations.jsonl \
  --output artifacts/conversations/teacher-traces.jsonl
```

- [ ] Add parser tests for all arguments and defaults. `--split` choices are `train`, `dev`, and `diagnostic_holdout`; defaults are 6/16/20 and `train`.

- [ ] Add a three-profile batch test where one completes, one reaches hard max, and one raises a provider error. Assert records go only to `conversations.jsonl`, `truncated-conversations.jsonl`, and `conversation-failures.jsonl`, respectively; the command continues to later profiles and returns exit code 1 when any profile fails.

- [ ] Assert `ConversationFailure` captures conversation/profile IDs, base seed, exact stage, error type/text, complete-round count, and checkpoint path. For adapter rejection before a valid ID exists, use the deterministic label `input-index-{zero_based_index}` for both profile identity and the conversation-ID prefix.

- [ ] Add resume tests:

  - finalized IDs found in either completed or truncated output are skipped without model calls;
  - an unfinished checkpoint is loaded and resumed;
  - output files already existing without `--resume` raise `FileExistsError`;
  - duplicate normalized profile IDs in one input raise `ValueError` before model calls;
  - retrying a prior failure is allowed, but finalized output is never duplicated.

- [ ] Add flatten tests asserting every round's nested `TeacherTrace` is emitted in input-conversation order and round order, and each row validates with existing `TeacherTrace`.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_cli.py
```

Expected: parser and dispatch failures because the commands do not exist.

- [ ] Register both commands. Instantiate one `OpenAIBackend` shared by `SeekerSimulator`, `TeacherRunner`, and `DialogueManager`; each component still resolves its role-specific `ModelConfig`. Use `MappingProfileAdapter` without interpreting profile fields.

- [ ] Before generation, read completed and truncated output ledgers into finalized ID sets. During generation, pass an atomic checkpoint callback. Append only a fully validated terminal trace to its status-specific ledger. Never construct a failed `ConversationTrace`.

- [ ] Implement `flatten-conversations` with `ConversationTrace.model_validate`, `flatten_teacher_traces`, and existing `write_jsonl`.

- [ ] Run:

```bash
.venv/bin/pytest -q tests/test_cli.py tests/test_storage.py tests/test_conversation.py
```

Expected: pass.

- [ ] Commit:

```bash
git add src/ibd/cli.py tests/test_cli.py
git commit -m "feat: add resumable conversation generation cli"
```

---

### Task 10: Prove information isolation and document pilot operation

**Files:**

- Modify: `tests/test_conversation.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`
- Modify: `configs/deepseek_teacher.yaml`

- [ ] Add an end-to-end privacy regression using the profile-only value `PRIVATE_PROFILE_SENTINEL`. The scripted seeker returns a normal utterance without the sentinel. Assert:

  - only `seeker_simulator` messages contain the sentinel;
  - manager, context updater, analyzer, planner, candidates, and selector messages do not;
  - public turns, all `TeacherTrace` fields, checkpoint JSON, completed/truncated JSONL, and flattened traces do not contain it;
  - supporter prompts contain only what the seeker actually disclosed.

- [ ] Add a schema regression that recursively inspects `ConversationTrace.model_json_schema()` and confirms there are no fields named `private_state`, `profile`, `profile_payload`, `dialogue_goal`, `should_close`, `supporter_traces`, or `dialogue_decisions`.

- [ ] Run privacy tests before documentation/config edits:

```bash
.venv/bin/pytest -q tests/test_conversation.py tests/test_cli.py -k "privacy or profile or sentinel or schema"
```

Expected: failures until all serialization and prompt boundaries are clean.

- [ ] Add `seeker_simulator` and `dialogue_manager` role entries to `configs/deepseek_teacher.yaml`. Keep the supporter roles unchanged. Use `provider_json_mode: false`, temperature `0.7`, max tokens `256` for seeker; use JSON mode, temperature `0.0`, max tokens `256` for manager.

- [ ] Expand README with:

  - one-profile/one-conversation architecture;
  - raw-profile visibility boundary;
  - dynamic mode semantics and defaults;
  - completed versus truncated versus failure artifacts;
  - generation, resume, and flatten commands;
  - warning that truncated traces are excluded from normal training input;
  - 50-profile pilot audit fields: completion/truncation/failure rates, length distribution, mode distribution/transition matrix, strategy distribution, repeated questions/suggestions, closing naturalness, and Context groundedness.

- [ ] Run the complete verification suite:

```bash
.venv/bin/pytest -q
```

Expected: all tests pass.

- [ ] Run CLI smoke checks without network/model calls:

```bash
.venv/bin/python -m ibd.cli generate-conversations --help
.venv/bin/python -m ibd.cli flatten-conversations --help
```

Expected: both commands print usage and exit 0.

- [ ] Inspect the final diff for accidental profile propagation and hard-max coercion:

```bash
rg -n "profile|hard_max|closing|dialogue_mode|transition_reason" src/ibd tests README.md configs/deepseek_teacher.yaml
git diff --check
git status --short
```

Expected: profile payload construction is confined to `seeker.py` and CLI adapter input; hard-max logic classifies rather than rewrites manager decisions; `git diff --check` is silent.

- [ ] Commit:

```bash
git add README.md configs/deepseek_teacher.yaml tests/test_conversation.py tests/test_cli.py
git commit -m "docs: document complete dialogue generation workflow"
```

---

## Final Acceptance Checklist

- [ ] One profile input produces one complete alternating conversation through one CLI batch item.
- [ ] Seeker alone sees the rendered profile; the architecture-level sentinel test passes.
- [ ] Every supporter round executes Context -> STATE -> Mode -> PLAN -> candidates -> selection.
- [ ] Manager output contains only mode and audit reason; Planner receives only mode as a weak prior.
- [ ] `MultiViewStateAnalysis` is reused exactly and `TeacherTrace.state == TeacherTrace.state_analysis.state` remains enforced.
- [ ] Normal completion occurs only after a closing-mode supporter response.
- [ ] Non-closing hard-max output is truncated and written outside completed training data.
- [ ] Every round binds one decision and one full Teacher trace.
- [ ] Resume restarts after the last complete round without replaying prior calls.
- [ ] Flattened rows validate as current `TeacherTrace` and remain consumable by Stage A/B.
- [ ] Existing single-target teacher, export, and training tests remain green.
- [ ] Profile schema decisions remain outside this implementation.
