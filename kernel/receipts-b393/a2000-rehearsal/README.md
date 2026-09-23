# B393 rehearsal on the NAS RTX A2000 — NOT the lane's reading

`census.json` is the second rehearsal of `kernel/b393_bitwise_census.py`, with the accuracy bound, disclosed in
`kernel/PREREG-b393-combine-reduce-bitwise.md`. It is **not a reading**:

- the card is an RTX A2000 (sm_86), not the RTX 5090 class the lane registers;
- the kernels are this branch's, not a pinned cut installed from a tag or merge commit;
- the image is `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.

It is kept because it changed the registration: its results showed that "at most 1 bf16 ULP" would have called
correct arithmetic a defect near cancellation. See the pre-registration's "What was run before this
registration was final". The lane's reading lands in `kernel/receipts-b393/5090/`.
