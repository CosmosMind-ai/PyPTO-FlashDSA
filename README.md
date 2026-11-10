# PyPTO-FlashDSA

**English** | [中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform](https://img.shields.io/badge/Platform-Ascend%20910C-blue)
![CANN](https://img.shields.io/badge/CANN-%E2%89%A5%209.0-orange)

> Part of the **CosmosMind PyPTO kernel series** for Ascend — all kernels driven by [MetaRSI](https://www.cosmosmind.ai/research/metarsi-v1).

DeepSeek Sparse Attention (DSA, **V4.1 semantics**) kernels for **Ascend 910C**, written
in [PyPTO](https://github.com/hw-native-sys/pypto) — the Ascend-native counterpart of
[DeepSeek-AI/FlashMLA](https://github.com/deepseek-ai/FlashMLA).

DSA (lightning indexer + top-k selection + FP8 sparse attention) is the sparse-attention
stack introduced by DeepSeek-V3.2 and finalized in V4.1. Just as DeepSeek gave MLA its own
library (FlashMLA), this repository gives DSA a dedicated, continuously-optimized kernel
library on Ascend.

## Status

v0.1.0 — **the first runnable V4.1 DSA implementation on Ascend 910C (`a2a3`)**.
The official Ascend C port is 950-only (it does not compile for 910C), and pypto-lib's
`deepseek_v4_1_flash` model path is likewise `a5`-only (per its own `ci_platforms.json`).
Every kernel in this repository is verified on physical 910C hardware.

## Components

| Component | Kernel | Holdout latency | Notes |
|---|---|---|---|
| Lightning indexer (scores) | `flash_dsa/kernels/lightning_indexer.py` | 726.29 µs | H=32, paged cache, causal/lengths |
| Lightning indexer (MXFP4 cache) | `flash_dsa/kernels/indexer_mxfp4.py` | 334.48 µs | MXFP4-quantized index path |
| Top-k selection (K=2048 semantics) | `flash_dsa/kernels/topk.py` | 62.15 µs | physical row ids, position-ordered, -1 padding, short-seq bypass |
| Sparse attention decode | `flash_dsa/kernels/sparse_attn_decode.py` | **54.45 µs** | 528B FP8 cache (e4m3+e8m0 g32), attn_sink, transposed lse, -1 invalid slots |

All holdout gains are +96% or better against the unfused same-DSL decomposition on the
evaluation shapes; see `benchmarks/BENCHMARKS.md` for protocols and per-shape numbers.

**The sparse-attention decode kernel is already faster than the official reference path**:
54.45 µs vs 56.49 µs for pypto-lib's official V4.1 fp32 reference path on the same card —
the only official equivalent that runs on 910C at all.

## Why DSA

DeepSeek's public position is that speculative decoding plus sparse attention deliver the
largest inference-side wins. Before this repository, ecosystem support was scattered
(cuDNN covers only certain SM100 paths; SGLang has a backend; FlashMLA has a prefill entry),
and nothing targeted Ascend. PyPTO-FlashDSA occupies that gap and keeps optimizing it.

## Installation

```bash
pip install -e .
```

Requirements: Ascend 910C (`a2a3`), CANN ≥ 9.0, `pypto`, `pypto-lib`, `torch`.

## Quickstart

```python
import torch
import flash_dsa

scores = flash_dsa.lightning_indexer(...)   # see flash_dsa/api.py and docs/SPEC.md
idx    = flash_dsa.topk(...)
out    = flash_dsa.sparse_attn_decode(...)
```

Component input/output semantics — including the paged KV-cache byte layout and the
physical-row-index conventions — are documented in `docs/SPEC.md`.

## Testing

```bash
python tests/test_kernels.py --device 0   # per-component torch-golden contracts (s-class shapes)
```

## Roadmap

- [ ] FlashMLA-compatible interface layer (`flash_mla_with_kvcache`, drop-in replacement)
- [ ] Port of the official test suite (FlashMLA `tests/`, incl. `quant.py` ue8m0/e2m1 semantics)
- [ ] vLLM-Ascend DSA path replacement (serving integration)
- [ ] Prefill path (decode shipped first)
- [ ] Continuous optimization on all components; numbers will keep moving

## Acknowledgements

V4.1 semantics follow FlashMLA master (V4.1-only upstream) and DeepSeek's official Ascend C
port (950-only, used as a semantic reference — it cannot run here). Correctness goldens are
torch references built to the official semantics. Every shipped kernel is gated by a sealed
correctness-and-latency holdout.

## License

MIT. See `LICENSE`.
