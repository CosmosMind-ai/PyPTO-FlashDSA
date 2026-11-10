# Benchmarks — PyPTO-FlashDSA

All numbers: Ascend 910C (`a2a3`), driver 26.0.rc1, median effective latency (µs),
100 rounds / 5 warmup, idle dies only. Raw evidence: `benchmarks/evidence/`.

## Components

| Component | Holdout latency | vs unfused same-DSL decomposition |
|---|---|---|
| Lightning indexer | 726.29 µs | +99.3% |
| Lightning indexer (MXFP4) | 334.48 µs | +99.5% |
| Top-k selection | 62.15 µs | +96.8% |
| Sparse attention decode | **54.45 µs** | +96.2% |

## vs the official reference path

pypto-lib's official V4.1 fp32 reference path is the only official equivalent that runs on
910C (the Ascend C port is 950-only). On the sparse-attention decode evaluation shape:

| | latency |
|---|---|
| Official fp32 reference path | 56.49 µs |
| PyPTO-FlashDSA sparse_attn_decode | **54.45 µs** |

i.e. the decode kernel is already faster than the official reference, while the official
*production* path for V4.1 does not target 910C at all — see the Status section of the
README for the platform-coverage evidence.

## Measurement discipline

- Every shipped kernel passes a sealed holdout with dual gates (accuracy + latency);
  candidates that are faster but incorrect are recorded as FAIL and never shipped.
- `benchmarks/evidence/`: `*.holdout.json` (raw holdout events) and `*.record.json`
  (decision-level measurement records).
- Optimization is still running on all four components; expect these numbers to move.
