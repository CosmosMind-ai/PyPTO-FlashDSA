"""Per-component contract tests — each kernel against its own torch golden on the
s-class evaluation shapes, on a real 910C device.

    python tests/test_kernels.py --device 0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_dsa.kernels import indexer_mxfp4, lightning_indexer, sparse_attn_decode, topk  # noqa: E402

S_SHAPES = {
    "lightning_indexer": (lightning_indexer, {"M": 64, "N": 1024}),
    "indexer_mxfp4": (indexer_mxfp4, {"M": 64, "N": 1024}),
    "topk": (topk, {"M": 64, "N": 4096}),
    "sparse_attn_decode": (sparse_attn_decode, {"B": 1, "S": 1, "N": 2048}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--device", type=int, default=0)
    args = ap.parse_args()

    from golden import run

    n_pass = 0
    for name, (mod, shape) in S_SHAPES.items():
        fn, specs, golden_fn, rtol, atol = mod.build(shape)
        result = run(fn=fn, specs=specs, golden_fn=golden_fn,
                     config=dict(backend_type="a2a3", device_id=args.device),
                     rtol=rtol, atol=atol)
        n_pass += bool(result.passed)
        print(f"{name:22s} {'PASS' if result.passed else 'FAIL: ' + str(result.error)}")
    print(f"\n{n_pass}/{len(S_SHAPES)} components passed")
    sys.exit(0 if n_pass == len(S_SHAPES) else 1)


if __name__ == "__main__":
    main()
