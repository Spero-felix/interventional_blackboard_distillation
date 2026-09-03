# Balanced teacher traces: design

## Objective

Create a new, auditable balanced derivative of the 2,500 teacher traces in
`artifacts/2500-30-45-25/teacher-traces.jsonl`. Keep the original prepared
data and trace file unchanged.

The derivative balances the final selected strategy across all 2,500 rows while
preserving example IDs, split membership, histories, and STATE labels.

## Final strategy targets

| Strategy | Current | Target | Target share |
| --- | ---: | ---: | ---: |
| Reflection of feelings | 1,029 | 592 | 23.68% |
| Question | 429 | 429 | 17.16% |
| Providing Suggestions | 379 | 379 | 15.16% |
| Affirmation and Reassurance | 324 | 324 | 12.96% |
| Restatement or Paraphrasing | 42 | 297 | 11.88% |
| Others | 279 | 279 | 11.16% |
| Information | 18 | 123 | 4.92% |
| Self-disclosure | 0 | 77 | 3.08% |

The transformation changes 437 traces currently selected as `Reflection of
feelings`: 105 become `Information`, 255 become `Restatement or
Paraphrasing`, and 77 become `Self-disclosure`.

## Split quotas

The 437 changes are stratified across the original split sizes rather than
concentrated in train.

| Split | Information | Restatement | Self-disclosure | Total changed |
| --- | ---: | ---: | ---: | ---: |
| train (2,000) | +83 | +208 | +62 | 353 |
| dev (200) | +9 | +20 | +6 | 35 |
| diagnostic_holdout (300) | +13 | +27 | +9 | 49 |
| Total | +105 | +255 | +77 | 437 |

This intentionally changes 84 evaluation rows. The original, unmodified trace
file remains the natural-distribution evaluation baseline; the balanced copy is
used only when a balanced evaluation or training condition is explicitly
required.

## Construction rules

1. Use deterministic, stratified sampling with seed `20260903`. Sample only
   currently selected Reflection rows and preserve split, conversation phase,
   dominant emotion, and distress-level representation as far as each quota
   permits.
2. Prefer already-generated, unselected target-strategy candidates when they
   are suitable after review. The current pool has 17 Information and 189
   Restatement alternatives among selected-Reflection rows.
3. Generate a replacement candidate when no reviewed reusable candidate is
   available: 88 Information, 66 Restatement, and 77 Self-disclosure replies.
4. A replacement must change the actual response, not only its label. Update
   the target candidate's strategy, response, response goal, and response act;
   update the strategy plan, final selection, and final response to the same
   canonical candidate. STATE and history must not change.
5. Information is allowed only for an explicit factual, conceptual, resource,
   or low-risk clarification need. Restatement must primarily paraphrase the
   seeker's content without converting it into emotional reflection or advice.
6. Self-disclosure must be brief, generic, low-risk, secondary to the seeker,
   and return focus to the seeker. It must not claim qualifications, diagnosis,
   treatment, medication, self-harm, abuse, crime, severe trauma, protected
   identity, specific personal history, or an externally verifiable event.
7. Candidate and final responses must be at most 30 tokenized words. Existing
   longer candidates are not eligible for reuse.

## Artifacts and provenance

Create a new artifact directory, not an in-place replacement. It contains:

- `teacher-traces.jsonl`: the balanced derivative;
- `manifest.json`: source paths, source hashes, deterministic seed, target and
  observed counts, changed example IDs, strategy transition counts, reuse vs.
  regeneration counts, and review status;
- `review_queue.jsonl`: rows requiring human semantic and Self-disclosure
  review, including old/new candidate metadata and response text;
- `failures.jsonl`: generation or validation failures, if any;
- an updated quality profile produced by the existing audit script.

The strict trace schema does not admit an extra provenance field, so all
derivative provenance lives in the manifest and review queue.

## Validation and acceptance criteria

1. `ibd.cli validate-trace` accepts all 2,500 balanced records.
2. IDs are unique and exactly match the original input; histories, split names,
   STATE labels, and phase metadata are unchanged per example ID.
3. Global selected-strategy counts exactly match the target table; per-split
   transition counts exactly match the quota table.
4. Every changed final response equals the selected candidate response; the
   candidate strategy matches its plan and final-selection fields.
5. All new/reused target candidates and final responses comply with the
   30-word limit. Existing out-of-scope length exceptions are reported but not
   silently altered by this balancing pass.
6. Human review approves every new Self-disclosure reply and a stratified
   sample of reused/re-generated Information and Restatement replies before the
   balanced artifact is used for training or evaluation.
7. The original artifacts remain byte-for-byte unchanged.

## Out of scope

- Rebalancing the original source artifact in place.
- Changing dialogue histories, STATE labels, or split membership.
- Claiming that the balanced evaluation set represents the natural deployment
  strategy distribution.
- Updating training configuration or launching a training run.
