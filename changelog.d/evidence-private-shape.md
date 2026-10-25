### Claims register: `evidence_private` names a private receipt the way a public reader may see it

`scripts/check_claims_register.py` (shared with experts4bit-qlora, byte-identical) now refuses an `evidence_private`
string that carries an absolute path, a URI, a receipts path under a checkout directory, a 7+ digit id, a UUID or a
public IPv4 address (`docs/claims-schema.md`, Private runs). Two rows here named their receipt under the receipt
store's checkout directory; they now name it by its run, `receipts/<repo>/<date>/<run>/`. The receipt's location is
unchanged and every other field of every row is byte-equal.
