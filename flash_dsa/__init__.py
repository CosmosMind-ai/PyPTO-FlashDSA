"""PyPTO-FlashDSA — DeepSeek Sparse Attention (V4.1) kernels for Ascend 910C."""

from .api import indexer_mxfp4, lightning_indexer, sparse_attn_decode, topk

__version__ = "0.1.0"
__all__ = ["lightning_indexer", "indexer_mxfp4", "topk", "sparse_attn_decode"]
