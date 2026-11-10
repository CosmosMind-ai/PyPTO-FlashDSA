# PyPTO-FlashDSA

[English](README.md) | **中文**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform](https://img.shields.io/badge/Platform-Ascend%20910C-blue)
![CANN](https://img.shields.io/badge/CANN-%E2%89%A5%209.0-orange)

> 本仓库属于 **CosmosMind PyPTO 算子系列**（面向昇腾），全部内核由 [MetaRSI](https://www.cosmosmind.ai/research/metarsi-v1) 驱动。

面向 **昇腾 910C** 的 DeepSeek Sparse Attention（DSA，**V4.1 语义**）内核库，基于
[PyPTO](https://github.com/hw-native-sys/pypto) 编写——是
[DeepSeek-AI/FlashMLA](https://github.com/deepseek-ai/FlashMLA) 在昇腾生态的对应实现。

DSA（lightning indexer + top-k 选择 + FP8 稀疏注意力）是 DeepSeek-V3.2 引入、
V4.1 定稿的稀疏注意力栈。正如 DeepSeek 为 MLA 单开了 FlashMLA，本仓库为 DSA 在
昇腾上提供一个专门的、持续优化的内核库。

## 状态

v0.1.0 —— **昇腾 910C（`a2a3`）上首个可运行的 V4.1 DSA 实现**。
官方 Ascend C 移植版仅支持 950（无法在 910C 编译），pypto-lib 的
`deepseek_v4_1_flash` 模型路径同样是 `a5` 专属（见其 `ci_platforms.json`）。
本库全部内核均在 910C 物理卡上验证。

## 组件

| 组件 | 内核 | Holdout 延迟 | 说明 |
|---|---|---|---|
| Lightning indexer（打分） | `flash_dsa/kernels/lightning_indexer.py` | 726.29 µs | H=32，分页 cache，causal/lengths |
| Lightning indexer（MXFP4 cache） | `flash_dsa/kernels/indexer_mxfp4.py` | 334.48 µs | MXFP4 量化索引路径 |
| Top-k 选择（K=2048 语义） | `flash_dsa/kernels/topk.py` | 62.15 µs | 物理行号、按位置排序、-1 填充、短序列旁路 |
| 稀疏注意力 decode | `flash_dsa/kernels/sparse_attn_decode.py` | **54.45 µs** | 528B FP8 cache（e4m3+e8m0 g32）、attn_sink、转置 lse、-1 无效槽 |

全部组件在评测形状上相对同 DSL 未融合分解的 holdout 收益均 ≥ +96%；
协议与逐形状数字见 `benchmarks/BENCHMARKS.md`。

**稀疏注意力 decode 内核已快过官方参考路径**：同卡 54.45 µs vs pypto-lib 官方
V4.1 fp32 参考路径 56.49 µs——后者是唯一能在 910C 上运行的官方等价实现。

## 为什么是 DSA

DeepSeek 的公开观点是：投机解码与稀疏注意力是推理侧收益最大的方向。本仓库之前，
生态支持零散（cuDNN 仅覆盖部分 SM100 路径；SGLang 有 backend；FlashMLA 有 prefill
入口），且没有任何面向昇腾的实现。PyPTO-FlashDSA 占据了这个空位，并持续优化。

## 安装

```bash
pip install -e .
```

环境要求：昇腾 910C（`a2a3`），CANN ≥ 9.0，`pypto`、`pypto-lib`、`torch`。

## 快速上手

```python
import torch
import flash_dsa

scores = flash_dsa.lightning_indexer(...)   # 见 flash_dsa/api.py 与 docs/SPEC.md
idx    = flash_dsa.topk(...)
out    = flash_dsa.sparse_attn_decode(...)
```

各组件的输入/输出语义——包括分页 KV-cache 字节布局与物理行索引约定——
见 `docs/SPEC.md`。

## 测试

```bash
python tests/test_kernels.py --device 0   # 逐组件 torch-golden 契约（s 级形状）
```

## 路线图

- [ ] FlashMLA 兼容接口层（`flash_mla_with_kvcache`，直接替换）
- [ ] 官方测试套件移植（FlashMLA `tests/`，含 `quant.py` 的 ue8m0/e2m1 语义）
- [ ] vLLM-Ascend DSA 路径替换（serving 集成）
- [ ] Prefill 路径（decode 先行）
- [ ] 全部组件持续优化，数字会继续推进

## 致谢

V4.1 语义遵循 FlashMLA master（V4.1 专用上游）与 DeepSeek 官方 Ascend C 移植版
（仅 950，作为语义参照——无法在本平台运行）。正确性 golden 为按官方语义构建的
torch 参考实现。每个发布的内核均通过密封的精度+延迟双门禁 holdout。

## 许可证

MIT，详见 `LICENSE`。
