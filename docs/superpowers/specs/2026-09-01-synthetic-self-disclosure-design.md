# Synthetic Self-disclosure Strategy Design

## Goal

Change the ESConv `Self-disclosure` strategy from non-autobiographical present
stance to brief synthetic supporter experience. Preserve the complete eight-class
strategy catalog while making the permitted fabrication narrow, generic, and
auditable.

## Strategy semantics

`Self-disclosure` may use a short first-person statement that presents a generic
supporter experience analogous to the seeker's situation, such as "I went through
something similar". The disclosure must remain secondary to the seeker's needs and
must return attention to the seeker within the same response.

The generated response does not disclose a real model experience. It is a synthetic
realization of the ESConv strategy used for data generation.

## Allowed synthetic experience

The Candidate may mention only common, low-risk, minimally detailed experiences or
emotional reactions. The statement must:

- be brief and generic;
- avoid names, dates, locations, institutions, or other verifiable details;
- avoid presenting the experience as evidence or authority;
- avoid implying that the seeker's outcome will match the supporter's claimed outcome;
- keep the main focus on the seeker.

Examples of the allowed form include:

- "I went through something similar, and it felt hard to know where to begin."
- "I've felt overwhelmed in a situation like that too."

These are examples of form, not text templates that must be copied.

## Prohibited synthetic experience

The Candidate must not invent:

- professional qualifications, clinical authority, or privileged expertise;
- diagnoses, treatment histories, medication use, or treatment outcomes;
- self-harm, suicide, abuse, crime, severe trauma, or other high-risk lived events;
- protected identity, family role, intimate relationship, employment history, or
  similarly specific biography;
- detailed or externally verifiable personal events.

The Candidate must not use a synthetic experience to pressure the seeker, prescribe
an outcome, claim certainty, or substitute the supporter's story for the seeker's
experience.

## Prompt synchronization

The shared ESConv catalog defines `Self-disclosure` using the synthetic-experience
semantics above. The Candidate prompt applies the same allowed and prohibited
boundaries only when `Self-disclosure` is assigned. General anti-fabrication rules
continue to protect facts about the seeker, dialogue, other people, events, and the
external world.

The Planner continues to treat `Self-disclosure` as one of eight strategies without
giving it an intrinsic priority. No STATE field maps directly to this strategy.

## Final Selector behavior

The Final Selector must not exclude a permitted generic synthetic supporter
experience solely because it is not factually verifiable. It still excludes a
Self-disclosure Candidate when the response:

- crosses a prohibited synthetic-experience boundary;
- makes unsupported claims about the seeker or external world;
- uses the supporter experience as authority or evidence;
- recenters the exchange on the supporter;
- materially conflicts with the dialogue or known STATE.

Conditional fit remains the first selection stage. Synthetic Self-disclosure receives
no special preference during response-quality comparison.

## Student prompt

The common Student-family system prompt remains strategy-neutral. It does not mention
Self-disclosure or synthetic experience. The behavior is learned from Teacher data
rather than imposed on every Student response.

## Protocol and cache isolation

Both Teacher configurations move to
`qwen25-socialsim-seven-state-policy-neutral-v2-synthetic-self-disclosure`. The new
protocol version creates a new Teacher cache namespace so earlier non-autobiographical
Self-disclosure outputs cannot be silently reused.

New Teacher traces and downstream intervention and anchor artifacts must be written to
a new artifact directory.

## Tests

Prompt tests require:

- the shared catalog to allow brief generic synthetic supporter experience;
- Candidate instructions to contain the low-risk boundary and remove the blanket ban
  on lived experience;
- Candidate instructions to retain anti-fabrication rules for seeker and external
  facts;
- Final Selector instructions to permit compliant synthetic Self-disclosure without
  treating it as inherently preferred;
- both Teacher configs to use the new protocol version;
- the Student system prompt to remain unchanged and strategy-neutral.

Static scans reject the superseded non-autobiographical Self-disclosure wording in
active runtime prompts.
