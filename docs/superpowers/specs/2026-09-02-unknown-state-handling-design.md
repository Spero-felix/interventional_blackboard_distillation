# Unknown STATE Handling Design

## Goal

Make `unknown` an epistemic missing-information value throughout the Teacher
protocol. It must not become an implicit user preference, a boundary, or a
default reason to prefer or reject a support strategy.

This change is strategy-neutral. It does not target a desired distribution of
`Reflection of feelings` or any other strategy.

## Central policy

Every prompt role that consumes STATE will receive the same policy:

> A known STATE value may affect only the local response effects assigned to
> that field. `unknown` means no usable evidence for that field. It must not be
> interpreted as any direction, preference, boundary, or default support
> action, and it must not count for or against a strategy or candidate.

The policy applies independently to every STATE field. It does not override an
explicit dialogue boundary or a known STATE value.

## Role-specific responsibilities

### State analyzer

- Keep the existing evidence contract: an `unknown` value requires
  `absent_or_ambiguous` evidence.
- Add the central policy in analysis terms: missing evidence must not be
  converted into an inferred user preference or support need.
- Preserve the distinction between a known value supported by explicit or
  strong-inference evidence and an unknown value.

### Planner

- Add the central policy before strategy applicability and ordering are
  evaluated.
- Select and order strategies from dialogue facts and known STATE values only.
- An unknown value must not supply positive or negative fit evidence for any
  category, including `Reflection of feelings`, `Question`, or `Providing
  Suggestions`.

### Candidate generator

- Add the central policy alongside the instruction to calibrate from known
  STATE fields.
- Unknown fields may not calibrate emotional wording, pace, directiveness,
  action burden, or conversational openness; only known field values may do
  so.
- The assigned strategy remains authoritative; this policy does not require a
  particular response form.

### Final selector

- Add the central policy in conditional-fit evaluation.
- Unknown values cannot advance, exclude, penalize, or otherwise rank a
  candidate. Conditional fit remains based on dialogue facts, explicit user
  boundaries, and known STATE values.

## Prompt structure

Define one shared prompt constant in `src/ibd/prompting.py` rather than
copying independently edited prose. Insert the same text in the four roles
above. Keep concise role-local sentences only where they clarify the role's
operation; do not duplicate a field-by-field policy.

The existing `advice_receptivity=unknown` guide definition remains unchanged:
it identifies absent attitude evidence. The central policy resolves its
downstream interpretation; it does not remove the enum value or force advice
to be given.

## Verification

1. Unit tests assert that every STATE-consuming role contains the shared
   unknown-handling policy.
2. Existing prompt tests continue to accept valid schemas and clamped-state
   prompts.
3. A small regenerated Teacher audit reports, for each STATE attribute value,
   its share, Reflection planning rate, and Reflection selection rate. Compare
   the audit to the current 400-example `-2` baseline without setting a target
   strategy proportion.
4. In that audit, inspect `advice_receptivity=unknown` against `open` within
   matched `primary_support_need` and conversation-phase strata. The criterion
   is semantic non-directionality in prompt behavior, not a preselected
   numerical win rate.

## Compatibility and scope

- This affects future Teacher traces and all downstream artifacts derived from
  them. Existing traces remain immutable historical artifacts and must not be
  mixed with regenerated outputs.
- No STATE enum changes are part of this design. The user's existing removal of
  `emotional_expression` is separate work and is intentionally left untouched.
- No training, intervention, or model-architecture changes are included.
