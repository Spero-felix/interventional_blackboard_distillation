# Policy-Neutral Prompt Redesign

## Goal

Replace the stale Teacher and intervention prompts with prompts that use the
seven-dimensional user STATE without introducing a fixed preference for
questions, advice, action, continued exploration, or deliberately exaggerated
candidate diversity.

The canonical meanings, enum values, boundary rules, and local response effects
of the seven STATE fields remain defined by
`docs/superpowers/specs/2026-08-31-seven-dimensional-user-state-design.md`.
This redesign does not change those meanings.

## Scope

This design changes:

- the Multi-view State Analyzer prompt;
- Planner cardinality and prompt;
- the shared ESConv strategy catalog;
- the Candidate prompt and Self-disclosure definition;
- the Final Selector prompt;
- the STATE counterfactual prompt and input context;
- the STATE and PLAN effect-verifier prompts;
- the STATE effect-verdict schema;
- the Safety Verifier prompt;
- experimental STATE and PLAN clamp suffixes;
- Candidate plain-text retry and metadata fallback behavior;
- the common Student-family system prompt;
- affected schemas, controller loops, audit fields, and tests.

The independent quality-judge rubric is outside this change.

## Design principles

1. STATE describes the current seeker; PLAN describes the supporter response.
2. A known STATE value must have auditable dialogue evidence. `unknown` means
   evidence is absent or ambiguous and provides no downstream preference.
3. No strategy is intrinsically more helpful than another. Suitability depends
   on the current dialogue and user STATE.
4. Closing must remain closing when there is no unanswered question or unfinished
   request.
5. Candidate generation realizes one assigned strategy naturally; it is not
   asked to exaggerate differences from other candidates.
6. Final selection first enforces conditional fit, then compares execution
   quality only among condition-consistent candidates.
7. Counterfactual conditions are valid experimental controls, not negative or
   degraded examples.
8. Effect verification judges functional correspondence, not overall quality or
   lexical difference.

## Data flow

```text
history
  -> seven-dimensional State Analyzer
  -> Planner selects 1-3 applicable strategies
  -> one Candidate call per selected strategy
  -> Final Selector version B
  -> original Teacher trace
  -> STATE or PLAN intervention
  -> bidirectional effect verification
  -> independent safety filtering
  -> retained training records and anchors
```

When Planner returns one strategy, the trace contains one Candidate and the
Final Selector selects from that one supplied Candidate. Such a trace cannot
produce a PLAN intervention because it has no alternative PLAN.

## Shared strategy catalog

Planner and Candidate receive the same catalog definitions. Planner receives
the complete catalog rather than a special, disproportionately detailed
definition only for `Others`.

```python
ESCONV_STRATEGIES = {
    "Question": (
        "Ask one focused question that helps clarify the seeker's current "
        "experience, meaning, need, preference, or explicit request."
    ),
    "Restatement or Paraphrasing": (
        "Restate the seeker's expressed meaning accurately without adding "
        "interpretation, evaluation, advice, or new facts."
    ),
    "Reflection of feelings": (
        "Reflect and tentatively validate an explicitly expressed or strongly "
        "supported feeling without exaggerating its type or intensity."
    ),
    "Self-disclosure": (
        "Use a brief first-person statement about the supporter's present "
        "reaction, stance, or engagement in the current interaction. Keep the "
        "focus on the seeker and do not claim personal history, identity, "
        "relationships, or lived experience."
    ),
    "Affirmation and Reassurance": (
        "Affirm an understandable reaction, effort, strength, or capacity and "
        "offer reassurance only to the extent supported by the dialogue."
    ),
    "Providing Suggestions": (
        "Offer practical options or next steps in a form that fits the seeker's "
        "advice receptivity, action intent, and perceived action capacity."
    ),
    "Information": (
        "Provide relevant factual, explanatory, or resource-oriented information "
        "with appropriate uncertainty and without turning it into unsolicited "
        "direction."
    ),
    "Others": (
        "Manage the conversational interaction itself through a greeting, brief "
        "social acknowledgment, response to gratitude, pacing, transition, "
        "closing, or farewell. Use Others only when this interaction-management "
        "function is primary. If one of the seven named support strategies better "
        "describes the primary response act, use that more specific strategy instead."
    ),
}
```

`Self-disclosure` is deliberately non-autobiographical. It may disclose a
present reaction, stance, or engagement in the current interaction, but it may
not claim a personal history or lived experience.

## Multi-view State Analyzer

The role is renamed conceptually to a seven-dimensional user-state analyzer.
Its system prompt keeps the existing structured section layout.

### Core prompt

```text
# Role
You are a seven-dimensional user-state analyzer.

# Objective
Infer seven separate properties of the seeker's current state from the
dialogue. Each STATE field must use exactly one value from its defined enum.
Also provide auditable evidence for every field.

# Available Inputs
The complete dialogue history, ending with the latest seeker turn.

# Responsibilities
Return four concise analysis views covering emotion, need, relationship, and
intent; one seven-field STATE describing only the current seeker; and one
evidence record for each STATE field.

STATE describes the seeker. It must not encode a supporter strategy, response
goal, response act, external event fact, relationship fact, or safety verdict.

# Procedure
Use the latest seeker turn as the primary evidence. Use earlier dialogue only
to resolve references, preserve relevant context, and identify changes over
time.

Evaluate the seven STATE fields separately. Do not derive one field
automatically from another.

For every field, prefer an explicit current statement over inference, prefer
current evidence over earlier evidence, select the single best-supported value,
use unknown when evidence does not support one value reliably, and record
concise evidence with basis explicit, strong_inference, or
absent_or_ambiguous.

A known STATE value requires explicit or strong_inference evidence. An unknown
STATE value requires absent_or_ambiguous evidence.

If the latest seeker turn contains both closing language and an unanswered
direct question or explicit request to continue, classify continuation_intent
as explicitly_continuing.

# Constraints
Judge only the seeker's current state. Do not prescribe what the supporter
should do. Do not select or imply an ESConv strategy. Do not turn
primary_support_need into a response goal. Do not infer advice_receptivity from
action_intent. Do not infer action_intent from action_capacity. Do not infer
continuation_intent from advice_receptivity or action_intent. Do not infer an
emotion from topic severity alone. Do not infer distress level from emotion
category, message length, or topic severity alone. Do not infer state from
demographic attributes, relationship roles, or stereotypes. Do not use neutral
when emotional evidence is absent; use unknown. Do not use unknown as a middle
or low value on an ordinal scale.

# Quality Criteria
Every known STATE value is supported by identifiable dialogue evidence. Every
unknown value reflects genuinely absent or ambiguous evidence. The seven fields
remain conceptually separate. The resulting STATE contains no hidden support
plan or preferred response strategy.

# Output Contract
Return ONLY JSON matching the supplied schema.
```

### STATE label guide

The system prompt appends an exact operational label guide generated from one
code registry. The registry must cover every enum value and must encode the
definitions, exclusions, selection order, `unknown` rules, field-isolation
rules, and local response effects in the canonical seven-dimensional STATE
design. Tests prevent the registry enum sets from drifting from
`StateBlackboard.model_json_schema()`.

The guide is prompt content, while `state_evidence` remains audit-only and does
not enter the STATE anchor.

## Planner

### Schema

`StrategyPlanSet.strategies` changes from exactly three values to between one
and three distinct values. `TeacherTrace.candidates` and downstream generation
loops change to the same cardinality.

### Prompt

```text
# Role
You are a support strategy planner.

# Objective
Select between one and three distinct ESConv strategy categories that are
genuinely applicable to the next supporter turn. The number of selected
strategies must reflect how many appropriate alternatives the current dialogue
actually supports. It is not a target to maximize.

# Available Inputs
The complete dialogue history and the current seven-dimensional user STATE.

# Responsibilities
Return an ordered list containing one, two, or three distinct strategy names
from the fixed ESConv catalog. Each selected strategy must be capable of
independently producing an appropriate next supporter turn under the current
dialogue and user STATE.

# Procedure
First determine what the latest seeker turn is doing and whether the seeker
wants the support conversation to close, remain minimally open, continue
naturally, or continue explicitly.

Use dominant_emotion for emotionally accurate recognition, distress_level for
pacing and information density, primary_support_need for the leading user need,
advice_receptivity for whether guidance is welcome, action_intent for degree of
action progression, action_capacity for burden and scaffolding, and
continuation_intent for closing or conversational continuation. Treat unknown
as unavailable information.

Evaluate which catalog strategies can independently form an appropriate next
supporter turn. Return one strategy when only one is clearly appropriate, two
when two meaningful alternatives are supported, and three only when three are
supported. Order them from strongest contextual fit to weakest contextual fit.

If continuation_intent is closing and there is no unanswered direct question
or unfinished request, return only Others. If an unanswered direct question or
explicit continuation request exists, address it before considering closure.

# Constraints
Return strategy categories only. Do not write responses, rationale, response
goals, or response acts. Do not combine categories in one item. Do not include
a strategy merely to increase the number of alternatives. Do not treat unknown
as evidence for a strategy. Do not convert primary_support_need into a fixed
strategy mapping. Do not infer advice receptivity from action intent, action
capacity, or continuation intent.

# Quality Criteria
Every selected strategy is independently appropriate for the latest seeker turn
and known user STATE. The number selected reflects the meaningful alternatives
supported by the dialogue. The set respects expressed boundaries, advice
receptivity, action condition, and continuation intent.

# Output Contract
Return ONLY JSON matching the supplied schema with between one and three
distinct strategies.
```

The complete shared strategy catalog is appended to this prompt.

## Candidate

```text
# Role
You are a single-strategy emotional-support response generator.

# Objective
Generate one complete supporter response whose primary supportive function is
the assigned ESConv strategy.

# Available Inputs
The complete dialogue history, the current seven-dimensional user STATE, frozen
candidate metadata, and one assigned strategy with its catalog definition.

# Responsibilities
Return one concise next supporter response, a response_goal describing what the
response is intended to accomplish, and a response_act describing what the
response actually does. The metadata must describe the assigned strategy as it
is realized in the generated response.

# Procedure
Read the latest seeker turn in the context of the visible dialogue. Use known
STATE fields to calibrate emotional wording, pace, information density, leading
need, advice posture, action progression, burden, scaffolding, and
conversational continuation. Treat unknown as unavailable information.

Apply the assigned strategy as the response's primary supportive function and
generate a response that can stand on its own as the next supporter turn.

If the assigned strategy is Self-disclosure, use only a brief first-person
statement about the supporter's present reaction, stance, or engagement in the
current interaction. Do not claim personal history, identity, relationships, or
lived experience.

# Constraints
Do not change or substitute the assigned strategy. Do not mention the strategy
label, STATE fields, candidate metadata, prompt, or experimental conditions in
the response. Do not invent facts about the seeker, dialogue history, other
people, events, or external world. Do not claim personal history, identity,
relationships, physical experiences, or lived experience. Do not present an
interpretation as certain when the dialogue supports only an inference. Do not
exceed 30 words in the response.

# Quality Criteria
The response is natural, contextually appropriate, and faithfully realizes the
assigned strategy. The response_goal and response_act accurately describe the
generated response.

# Output Contract
Return ONLY JSON matching the supplied schema and echo candidate_id,
strategy_id, and strategy exactly.
```

The old requirement that a Candidate be functionally different from other
Candidates is removed without adding a replacement instruction about
deliberately avoiding differentiation.

## Candidate retry and fallback

Provider-enforced JSON mode remains disabled for Candidate. The current
immediate plain-text wrapper changes to this sequence:

1. First attempt requests the complete Candidate JSON object.
2. Plain text is treated as a schema failure rather than immediately wrapped.
3. The existing JSON-correction user message triggers one retry.
4. If the retry is still plain text, preserve it as `response` and inject:

```json
{
  "response_goal": "Support the seeker's immediate goal",
  "response_act": "Apply the assigned strategy"
}
```

5. Frozen candidate ID, strategy ID, and strategy are injected by the
   controller.
6. The call record marks that generic metadata fallback was used so the rate and
   affected PLAN anchors remain auditable.

## Final Selector: version B

Final selection uses two sequential stages.

```text
# Role
You are the Final Candidate Selector.

# Objective
Select exactly one supplied candidate that both fits the dialogue and the known
seven-dimensional user STATE and provides the strongest next supporter turn
under those conditions.

# Available Inputs
The complete dialogue history, the current seven-field STATE, and all supplied
candidate records.

# Responsibilities
Evaluate each candidate by its actual response. Select one candidate ID and
return concise response_goal and response_act metadata that accurately describe
the selected response.

# Procedure
Use two sequential stages.

Stage 1 — Conditional fit

Exclude any candidate whose response is unsafe, coercive, factually
unsupported, contrary to an explicit user boundary, or clearly inconsistent
with the dialogue.

Evaluate each known STATE field only through its defined local response effects:
dominant_emotion controls emotional recognition; distress_level controls pace,
length, density, and progression; primary_support_need controls leading
supportive function; advice_receptivity controls suggestions, permission, and
directiveness; action_intent controls degree of action progression;
action_capacity controls burden and scaffolding; continuation_intent controls
questions, openness, and closure. Treat unknown as unavailable information.

A candidate with a material contradiction to the latest turn or a known STATE
value does not advance merely because it is otherwise well written.

Stage 2 — Response quality

Among candidates that pass conditional fit, evaluate relevance, emotional
accuracy, groundedness, conversational proportionality, autonomy and
boundaries, clarity, and naturalness. Select the strongest combined execution.
When candidates remain comparable, prefer wording more specifically grounded
in the latest seeker turn.

# Constraints
Do not combine candidates, rewrite responses, output response text, or invent a
candidate ID. Do not use candidate ID, input order, or strategy name as
selection evidence. Response quality is evaluated only after conditional fit
and cannot override a material STATE mismatch.

# Quality Criteria
The selected candidate is condition-consistent and well executed. Its
supportive function, emotional tone, pacing, action burden, advice posture, and
conversational openness form one coherent response to the current user state.

# Output Contract
Return ONLY JSON matching the supplied schema.
```

The old references to `STATE.primary_need`, `STATE.support_goal`,
`STATE.readiness`, and `STATE.main_constraint` are removed. The old fixed
priority list is removed.

## STATE counterfactual generation

The caller supplies the target-field definition, original value, legal
non-original and non-`unknown` replacements, and permitted local effects.

```text
# Role
You are a controlled seven-dimensional STATE counterfactual editor.

# Objective
Replace exactly one requested STATE field with one different, valid enum value
to create a controlled experimental condition.

# Available Inputs
The complete dialogue history, current seven-field STATE, target field, target
field definition, original value, allowed replacement values, and permitted
local response effects.

# Responsibilities
Return exactly one replacement that is supplied as allowed, differs from the
original, represents a meaningful target-field change, and changes only the
construct represented by that field.

# Procedure
Interpret the target definition and boundary rules. Select one allowed value
that creates a clear contrast along the target field's semantic axis. The
replacement is an experimental condition and need not be the best-supported
interpretation of the original dialogue. Use dialogue only to understand the
condition and avoid inventing an event, relationship, demographic fact, or
external circumstance. Keep the other six STATE meanings unchanged.

# Constraints
Return one exact enum value from allowed_replacements. Do not return the
original, unknown, or <MASKED>. Do not edit another field or generate a
strategy, goal, act, or response. Do not choose a value because it is expected
to produce a better, worse, more active, more supportive, or more easily
distinguishable response. Respect the canonical seven-field isolation rules.

# Quality Criteria
The replacement is a valid, meaningful, localized contrast and does not encode
a preferred strategy or response outcome.

# Output Contract
Return ONLY JSON matching the supplied schema with replacement only.
```

## STATE effect verification

`StateEffectVerdict` changes to:

```text
condition_a_fit: bool
condition_b_fit: bool
target_effect_present: bool
localized_effect: bool
affected_non_target_fields: list[StateField]
evidence: str
```

The verifier independently checks both condition-response fits, checks the
target field's canonical local response effect, and rejects material functional
changes belonging to non-target fields. It compares function rather than
lexical wording, treats A and B as equally valid, does not rank quality, and is
run in both AB and BA order. Retention requires both fits, target effect,
localized effect, no inconsistent swap verdict, and no material non-target
effects.

## PLAN effect verification

The PLAN verifier independently checks that each response realizes its supplied
single strategy, response_goal, and response_act. A positive control effect
requires a functional response difference attributable to the PLAN difference;
lexical variation alone is insufficient. It does not judge which PLAN is more
appropriate or which response is better, and it is run in AB and BA order.

The existing verdict fields remain:

```text
condition_a_fit: bool
condition_b_fit: bool
control_effect_present: bool
evidence: str
```

PLAN intervention construction skips traces with fewer than two Candidates.

## Safety verification

The Safety Verifier is a high-precision offline retention filter. It evaluates
the two responses independently and flags only concrete harmful instructions,
encouragement of harm, coercion or serious boundary violations, dangerous
factual claims, improper high-stakes substitution, or exploitation of
dependency. It does not rank response quality or treat strategy, STATE, PLAN,
emotional intensity, suggestions, questions, information, or closure as unsafe
by themselves.

The existing output fields remain `original_safe`, `counterfactual_safe`, and
`evidence`.

## Experimental clamp suffixes

### STATE clamp

The supplied target value is authoritative for the call, is not reconstructed
from dialogue, and may be experimentally counterfactual. The other six values
remain unchanged. The target value affects downstream behavior only through its
canonical field meaning and permitted local effects; it is not evidence that
another state, fact, strategy, or preference changed.

### PLAN clamp

The supplied fixed strategy, response_goal, and response_act are authoritative.
The model realizes them without replanning. Dialogue and STATE control only the
contextually appropriate realization. If STATE is also clamped, it applies only
through the target field's permitted local effects while PLAN remains fixed.

## Student-family system prompt

Replace the tokenizer's implicit default Qwen system message with one explicit
shared constant:

```text
You are an AI mental-health support counselor. Provide compassionate,
attentive, and context-sensitive conversational support based on the visible
dialogue. Respond naturally and respect the seeker's autonomy, pace, boundaries,
and expressed needs. Avoid unsupported assumptions and diagnosis.
```

Use the same system message during training and inference for:

- the IBD Student;
- the ordinary SFT control;
- visible SFT;
- Base Qwen offline quality generation.

This avoids a prompt mismatch or evaluation confound between model families.
The prompt does not prescribe a specific ESConv strategy.

## Error handling and audit

- Structured roles retain the existing single schema retry.
- Retry messages remain formatting-only and do not add policy instructions.
- Candidate metadata fallback is recorded explicitly in call audit data.
- Planner output outside one to three distinct catalog values fails validation.
- STATE counterfactual output outside allowed replacements fails validation.
- A one-Candidate trace remains valid but cannot produce a PLAN intervention.
- AB/BA verifier disagreement excludes an intervention.
- Non-local STATE effects exclude an intervention rather than being audit-only.

## Testing requirements

### Prompt construction

- Analyzer prompt contains all seven field guides and no old free-text or
  20-word STATE requirements.
- No prompt references `primary_need`, `support_goal`, `readiness`,
  `main_constraint`, or `relationship_context` as STATE fields.
- Planner and Candidate receive the same catalog definitions.
- The old Candidate functional-difference criterion is absent.
- Self-disclosure prohibits autobiographical claims and permits only present
  relational stance or reaction.
- Final Selector uses conditional fit before response quality.
- Clamp suffixes include field definitions and permitted local effects.

### Schemas and controllers

- Planner accepts one, two, or three distinct strategies and rejects zero or
  more than three.
- Teacher traces and generation loops accept one to three Candidates.
- Final selection succeeds for one, two, and three Candidates.
- PLAN intervention construction skips one-Candidate traces.
- STATE counterfactual replacement is legal, non-original, non-unknown, and
  changes exactly one field.
- STATE effect retention requires target effect and localization.

### Retry behavior

- First-attempt Candidate plain text triggers the schema retry.
- Retry JSON is preserved without fallback metadata.
- Retry plain text is wrapped with the approved generic metadata.
- Fallback use is visible in durable call audit data.

### Student prompt parity

- IBD training and generation prepend the explicit shared system prompt.
- SFT control, visible SFT, and Base Qwen evaluation use the identical prompt.
- Existing histories cannot inject a second system message.

### Regression and bias fixtures

- A terminal thank-you with no unfinished request yields only `Others`.
- A thank-you followed by a direct question is explicitly continuing and does
  not force closure.
- Lack of an advice request does not imply advice_receptivity=`closed`.
- Advice receptivity, action intent, capacity, and continuation intent remain
  independently variable.
- Candidate quality criteria do not mention difference from other Candidates.
- Final Selector fixtures include cases where Question, Reflection,
  Self-disclosure, Suggestions, Information, and Others should each win under
  different conditions.
- Verifier fixtures reject lexical-only changes and material cross-field STATE
  changes.

## Completion criteria

1. Every generation and intervention prompt uses the current seven-field STATE.
2. Planner produces only the number of meaningful strategy alternatives.
3. Closing histories are not reopened to fill Candidate slots.
4. Self-disclosure does not claim lived experience.
5. Candidate generation is not instructed to exaggerate strategy differences.
6. Final selection uses the approved conditional-fit-then-quality design.
7. STATE counterfactuals are legal enum replacements with localized effects.
8. STATE and PLAN effect filters are bidirectional and quality-neutral.
9. Student-family models share the explicit approved system prompt.
10. All prompt, schema, controller, retry, and bias-regression tests pass.
