"""Python API for PyPTO-FlashDSA.

v0.1.0 exposes component-level entry points. Each takes a shape dict and a
dict of input tensors (names per docs/SPEC.md), binds them onto the kernel's
spec list, and launches through the public `golden` harness (pypto-lib) with
`save_data=True` — the same validated path the kernels were verified with,
at the cost of a host-side tensor round-trip per call. A zero-overhead direct
launcher and the FlashMLA-compatible layer are on the roadmap.
"""
import tempfile
from pathlib import Path

import torch

from .kernels import indexer_mxfp4 as _mxfp4
from .kernels import lightning_indexer as _indexer
from .kernels import sparse_attn_decode as _decode
from .kernels import topk as _topk

_KERNELS = {
    "lightning_indexer": (_indexer, ["score"]),
    "indexer_mxfp4": (_mxfp4, ["score"]),
    "topk": (_topk, ["indices"]),
    "sparse_attn_decode": (_decode, ["out", "lse"]),
}


def _run(name, shape, inputs, device, validate):
    mod, out_names = _KERNELS[name]
    fn, specs, golden_fn, rtol, atol = mod.build(shape)
    for spec in specs:
        if spec.name in inputs:
            t = inputs[spec.name]
            if list(spec.shape) != list(t.shape):
                raise ValueError(f"{name}: {spec.name} expects {spec.shape}, got {list(t.shape)}")
            spec.init_value = t.contiguous()
    from golden import run
    work = Path(tempfile.mkdtemp(prefix=f"flashdsa_{name}_"))
    result = run(fn=fn, specs=specs, golden_fn=golden_fn if validate else None,
                 config=dict(backend_type="a2a3", device_id=device),
                 rtol=rtol, atol=atol, save_data=True)
    if validate and not result.passed:
        raise RuntimeError(result.error or "golden validation failed")
    outs = []
    for out in out_names:
        path = work / "data" / "out" / f"{out}.pt"
        if not path.exists():
            raise RuntimeError(f"output {out!r} not produced (missing {path})")
        outs.append(torch.load(path, weights_only=True))
    return outs if len(outs) > 1 else outs[0]


def lightning_indexer(shape, inputs, device=0, validate=False):
    """V4.1 lightning-indexer scores. See docs/SPEC.md for tensor semantics."""
    return _run("lightning_indexer", shape, inputs, device, validate)


def indexer_mxfp4(shape, inputs, device=0, validate=False):
    """Lightning-indexer scores over an MXFP4-quantized cache."""
    return _run("indexer_mxfp4", shape, inputs, device, validate)


def topk(shape, inputs, device=0, validate=False):
    """Top-k index selection (K=2048 semantics, physical row ids, -1 padding)."""
    return _run("topk", shape, inputs, device, validate)


def sparse_attn_decode(shape, inputs, device=0, validate=False):
    """Sparse attention decode over the packed FP8 (e4m3+e8m0 g32) cache."""
    return _run("sparse_attn_decode", shape, inputs, device, validate)
