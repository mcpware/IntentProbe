# Memory Label Policy

## Separate decisions

### Write decision

- `SAVE`: durable storage serves a user-directed future purpose and is permitted by the applicable
  privacy/preferences policy.
- `SKIP`: use only for the current interaction; no durable persistence.

### Delete decision

- `FORGET`: a stored item matches a valid deletion instruction.
- `KEEP`: the deletion request does not identify this item.

### Retrieval decision

- `RETRIEVE_ALLOWED`: the current request and context authorize using the stored item.
- `RETRIEVE_DENIED`: the item may exist but must not be surfaced in this context.

These heads cannot be substituted for one another.

## Minimum example fields

- current utterance and relevant conversation;
- speaker/subject ownership, including third-party data;
- sensitivity category;
- temporal scope and expiry;
- explicit save/no-save/forget instruction;
- durable-preference versus temporary-state distinction;
- source provenance and label confidence;
- seed/user/scenario group IDs.

## High-value contrast families

- “Remember this for next time” versus “Use this only for this task”.
- Stable dietary/accessibility preference versus today's temporary choice.
- User's own information versus quoted third-party information.
- Exact secret versus a safe fact that a secret exists.
- Preference update versus accumulating contradictory versions.
- Save authorization versus retrieval authorization in an unrelated context.
- Explicit deletion versus a new preference that merely supersedes the old one.

## Release rule

No real conversation enters a public dataset or released weight without documented provenance,
applicable permission, de-identification review, and a reason the intended use is compatible with the
source terms.
