# SocialSim Phase-Stratified Sampling Design

## Problem

`prepare-socialsim` currently converts each eligible SSConv conversation into the
history immediately preceding its last observed Supporter turn. Conversation
selection is randomized, but the response target inside every selected
conversation is fixed at the end. In the existing 2,000-example artifact, all
2,000 examples therefore fall in the late phase rather than representing the
middle of the dialogue.

The prepared dataset must instead contain approximately 10% early, 80% middle,
and 10% late contexts. The split boundary must remain conversation-safe, the
selection must remain reproducible, and no target or future Supporter text may
appear in an exported history.

## Selected Approach

Use phase-stratified sampling independently inside each output split.

For each valid source conversation:

1. Parse and validate the complete alternating Seeker/Supporter sequence.
2. Treat every Supporter turn as a possible response target. The exported
   history is the prefix ending at the Seeker turn immediately before that
   target.
3. Divide the ordered target positions into three contiguous, near-equal bins
   using their ordinal ranks: early, middle, and late.
4. Assign selected conversations to phase quotas within each split, shuffle the
   assignments deterministically, and choose one target uniformly from the
   assigned phase.

The default quotas are 10% early, 80% middle, and 10% late. Integer quotas use a
deterministic largest-remainder allocation so that every example is assigned
exactly once. With the production split sizes of 1,500/200/300, the result is
exactly 150/1,200/150, 20/160/20, and 30/240/30.

This is preferred to weighted sampling, which only guarantees the distribution
in expectation, and to a narrow fixed center window, which reduces context
diversity.

## Data Contract

New artifacts use protocol version `socialsim-qwen-conversation-v2`. Each
`SocialSimExample` records:

- `conversation_phase`: `early`, `middle`, or `late`;
- `target_turn`: the one-based source Supporter turn being predicted;
- `target_rank`: the one-based ordinal rank of that target among all valid
  Supporter targets;
- `eligible_target_count`: the total number of target positions in the source
  conversation.

The manifest records requested phase fractions and phase counts globally and by
split. Version-1 artifacts remain readable. Their phase metadata is absent
rather than fabricated, because the old artifact loader must not claim that it
audited information it never stored.

New preparation excludes a conversation if it has fewer than three target
positions, because such a conversation cannot supply non-empty early, middle,
and late bins. The exclusion reason is recorded in the existing manifest
`excluded` mapping.

## CLI

`prepare-socialsim` gains two optional arguments:

- `--early-fraction`, default `0.1`;
- `--late-fraction`, default `0.1`.

The middle fraction is derived as `1 - early - late`. Fractions must be finite,
non-negative, and leave a positive middle fraction. Defaults implement the
approved 10/80/10 protocol while retaining an explicit, reproducible experiment
surface.

## Reproducibility and Leakage Safety

Conversation records are sorted by ID before any seeded shuffle, preserving
input-order independence. Separate deterministic random streams are derived for
conversation selection and each split's phase/target selection, so changing one
split size cannot silently perturb target choices in earlier splits.

Only turns strictly before the selected Supporter target are exported. The
target response, its hidden reasoning fields, and every later turn are omitted.
Profile data continues to be used only for ID-integrity validation.

## Downstream Collapse Investigation

The sampling fix is isolated from Teacher and Student objective changes. The
existing 2,000-example artifacts are inspected at each downstream boundary:

1. prepared histories to Teacher plans;
2. plans to final candidate selections;
3. Teacher traces to accepted interventions;
4. intervention function balance to Stage C batches;
5. training objectives and dev checkpoint selection.

The report will distinguish measured collapse signals from mechanisms that are
only risks. No Teacher prompt, intervention acceptance rule, or training loss is
changed until regenerated phase-balanced traces show whether the bias persists.

## Tests

Tests must demonstrate:

- exact 10/80/10 phase counts for production-compatible split sizes;
- histories end at the Seeker immediately before the chosen target;
- target and future Supporter responses/reasoning never leak;
- target ranks fall inside the recorded phase bin;
- results are identical when source input order is reversed;
- invalid fractions and conversations with fewer than three targets are handled
  explicitly;
- the CLI writes a version-2 artifact with auditable global and per-split phase
  counts;
- legacy version-1 prepared artifacts remain accepted by Teacher input loading.

## Non-Goals

- Reusing or overwriting the existing 2,000-example artifact.
- Regenerating Teacher traces or interventions during this change.
- Rebalancing Teacher strategies without evidence from newly generated data.
- Changing Stage A/B/C optimization in the same patch.
