### Claims register: retired rows can say when the claim was published

A `retired` row may now carry `published_on: {date, evidence}`, the first public appearance of the claim it retracts,
so `published_on` to `retired_on` is how long a wrong claim stood. The evidence is `<commit>:<path>`, which the check
resolves: the commit is in this repository's history, the path is in its tree, and the date is its committer date. A
URL is also accepted; it is listed and never resolved. The field is refused on any other status and on a date after
`retired_on`. No row here is backfilled; it is unset unless known. `docs/claims-schema.md`,
`scripts/check_claims_register.py` and its test change in lockstep with experts4bit-qlora.
