# LIDU provenance

Original implementation: `mtp_xzw_0903` at
`96dff93d3eb98acc66e8fcd405a1c6b468aef8a7`, directory
`csrc/attention/nanovllm_a5_ops/a5_lidu_fused_batch_get_prepare_mtp`.

This copy starts from the restored original implementation already present in
`codex/kvaf-dynamic-hot-cache` at `679d69cd14969404fba31df95fb51f456a740378`.
It retains that copy's capacity/bitmap/scratch-size adaptations and Torch stream
launcher fix. Resident capacity is 133632 slots, including spare slots.

This integration changes only these query-layout validation rules:

- Admit 1..4 queries rather than 2..4, using the existing per-query loop.
- Admit zero queries only for the existing native padding-row sentinel.
- Permit a trailing unused suffix of the captured TopK tensor.

Classification, compaction, deduplication, victim selection, map mutation and
descriptor generation are unchanged from that restored implementation.
The Numba LRU kernels are separately preserved in `../kvop/lru_core.py`.

In retained-HBM mode, the adapter supplies 2048 empty candidate slots, not
occupied LRU victims. It rejects any nonzero/negative miss feedback before LRU
commit. Get is not invoked; SFA continues to consume the original LI indices.
