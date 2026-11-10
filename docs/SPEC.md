# PyPTO-FlashDSA — V4.1 Semantics and Tensor Conventions

Reference sources: FlashMLA master (V4.1-only upstream) and DeepSeek's official Ascend C
port (950-only; semantic reference). Correctness goldens are torch references built to
these semantics.

## Pipeline overview

DSA decode runs three stages: the **lightning indexer** scores every cached page against
the current query heads; **top-k** selects the K=2048 (semantics) most relevant physical
cache rows; **sparse attention decode** attends only over the selected rows against the
packed FP8 KV cache.

## lightning_indexer

Tensors: `q` [M, H*D] bf16 · `cache` [NB, PAGE, D] bf16 · `block_table` [1, P] int32 ·
`lengths` [M] int32 · `w` [M, H] fp32 → `score` [M, N] fp32 (`dots` is an internal
staging buffer).

H=32 heads, D=128, PAGE=128. Causal masking per `lengths`.

## indexer_mxfp4

Same scoring role over an MXFP4-quantized index cache: `cache` (packed e2m1), `scales`
(ue8m0), `pidx`/`sidx` index tensors → `score`.

## topk

Tensors: `scores` [M, N] fp32 · `lengths` [M] int32 · `block_table` · `cutoffs`/`quotas`/
`wpairs` (control-plane tensors) → `indices` — physical cache-row ids, position-ordered,
-1-padded, with a short-sequence bypass.

## sparse_attn_decode

Tensors: `query` · `cache` (packed 528-byte FP8 rows: e4m3 values + e8m0 group-of-32
scales) · `indices` (from topk, -1 = invalid slot, skipped) · `sink` (attention sink) →
`out` + `lse` (transposed layout). Rows whose indices are all invalid follow the
documented boundary semantics.

## Evaluation shapes

| class | lightning_indexer / indexer_mxfp4 | topk | sparse_attn_decode |
|---|---|---|---|
| s | M=64, N=1024 | M=64, N=4096 | B=1, S=1, N=2048 |
| m | M=512, N=8192 | M=512, N=8192 | B=1, S=2, N=16384 |
| l | M=256, N=32768 | M=256, N=32768 | B=2, S=4, N=65536 |
