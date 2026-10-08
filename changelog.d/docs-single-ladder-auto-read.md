### Docs: `NF4_QLORA_SINGLE_LADDER=auto` on a second host

STATUS records experts4bit-qlora TC1 amendment 71: `auto` laddered only the fp32-adapter arm and held its mechanism, but
the step time went unread on a loaded host. It stays opt-in until amendment 72 reads it on a third host.
