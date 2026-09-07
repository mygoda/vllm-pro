# 适配与优化详情

记录 vllm-pro 相对上游做的模型/硬件适配，以及每个版本的性能数据。

---

## GLM-5.3-Flash × RTX 4090 (sm_89)

**模型**: GLM-5.3-Flash（320B-A18B MoE，native FP8）  
**硬件**: 8× RTX 4090（sm_89 / Ada），24.5 GB/卡，共 196 GB VRAM  
**日期**: 2026-09-01

### 背景

上游 vLLM 与 SGLang 在 sm_89 上均无法正确运行该模型（输出乱码）。sm_89 缺少 Hopper 专属指令（TMA、wgmma、FP4 张量核），导致 DeepGEMM、FlashMLA-Sparse 等关键内核全部 gate 在 Hopper/Blackwell，无任何 fallback。

### 核心问题与修复

**根因：mHC 融合 RMSNorm 缺失**

GLM-5.3 将 input_layernorm 的 RMSNorm 融合进 mHC（multi-head hyper-connection）pre kernel。tilelang 版正确实现了 `norm_weight` 参数；但 native torch 版 `mhc_pre_torch` 根本没有该参数，`forward_native` 接收后静默丢弃。sm_89 上为绕开 tvm_ffi 崩溃强制走 native 路径，导致**每层 layernorm 学习权重全部丢失**，attention/MLP 收到幅度差 13× 的未归一化输入 → 乱码。

修复：3 处改动，`mhc_pre_torch` 补全 `norm_weight`/`norm_eps` 参数并正确施加 RMSNorm，透传到 `forward_native` 和 `MHCFusedPostPreOp`。修复后 top-token logprob 从 -5 → -0.5。

**新增 sm_89 专用内核**

| 内核 | 文件 | 说明 |
|------|------|------|
| sparse-MLA 后端 | `vllm/v1/attention/backends/mla/torchref_mla_sparse_sm89.py` | gather topk KV + torch einsum，NoPE-aware，数学对拍 REL=0.0014 |
| DSA logits（prefill） | `vllm/utils/deep_gemm.py` | `_sm89_torch_ref_mqa_logits`，替代 Hopper-only deepgemm |
| DSA logits（paged decode） | `vllm/utils/deep_gemm.py` | `_sm89_torch_ref_paged_mqa_logits`，全向量化以支持 CUDA Graph capture |

**CUDA Graph 解锁**

paged DSA fallback 写成完全向量化（无 `.item()`/python 循环）→ 可被 CUDA Graph capture。配合 `cudagraph_capture_sizes=[1,2,4,8]` + `expandable_segments:True`。

### 性能结果

| 配置 | decode 速度 | TTFT | 备注 |
|------|------------|------|------|
| eager（初版） | 3.5 tok/s | 0.50 s | 基线 |
| CUDA Graph | 5.1 tok/s | 0.37 s | +46% |
| **CUDA Graph + KV/权重平衡（最终）** | **7.16 tok/s** | — | **最优，已锁定** |

最终配置：`cpu-offload-gb=24`（驻留更多权重）+ `--kv-cache-memory 4294967296`（KV 砍至 4 GB）。降 offload 40→24 多驻留约 14 GB/卡权重，消除流式瓶颈。`offload-18` 即 OOM，当前已到 4090 天花板。

**最终 7.16 tok/s = 初始 eager 的 2.05×。**

### 已知限制

- 长 prompt prefill（>2048 token）约 18 s：走 torch DSA fallback（全窗口 einsum），可写 triton 版提速
- KDA / 自写 MLA kernel 是 eager_break，暂不在 CUDA Graph 覆盖范围内
- 仅验证到 8192 上下文

详细调试过程见 [GLM5.3-SM89-优化报告.md](../GLM5.3-SM89-优化报告.md)。

### 进行中 / 待验证的优化

以下改动已落地代码，需在 8×4090 服务器上实测确认收益：

| 项 | 类型 | 说明 | 状态 |
|----|------|------|------|
| prefill DSA logits 按 M 分块 | 代码 | `_sm89_torch_ref_mqa_logits` 原本一次性 materialize `[H,M,N]` fp32 score（M=N=4096 时约 2 GiB），改为按 512 行分块、循环内 reduce over H，峰值降到约 256 MiB。数学等价，附 CPU 自检 `tests/kernels/attention/test_sm89_dsa_logits.py` | ✅ 已改，待实测 prefill 提速 |
| prefill DSA logits Triton 融合 kernel | 代码 | `vllm/utils/sm89_dsa_triton.py`：`qk→relu→加权→reduce_h→mask` 融进单 kernel，中间 score 不落 HBM。默认 fp32（`input_precision="ieee"`）匹配参考精度；`VLLM_SM89_DSA_LOGITS_BF16=1` 切 bf16 dot 走 Ada tensor core（fp32 累加），快数倍、精度换 topk 排序（DSA logits 只喂 topk 选择）。GPU 上自动启用，CPU/超大 head dim 回退分块 torch 版 | ✅ 已改，待服务器 A/B |
| `--moe-backend` 核查 | 配置 | GLM-5.3 是 native FP8 MoE，当前启动脚本用 `marlin`（W4A16 GPTQ 专用），应改 `triton` 或 auto | ⏳ 服务器侧待试 |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | 配置 | 34 个 KDA 层现走 eager；breakable 图模式可把 `_forward` 当 eager segment、capture 前后投影，若 recurrent kernel 可 capture 则 decode 提速 | ⏳ 服务器侧待试 |
| `--mamba-cache-mode align` | 配置 | **KDA prefix caching 零代码启用** —— vLLM 已内置 `MambaManager` align 模式，默认 `none`。多轮/长系统提示可省整段 KDA prefill。详见 [kda-prefix-cache.md](kda-prefix-cache.md) | ⏳ 服务器侧待试 |
| topk 寄存器驻留 Triton kernel | 代码 | `vllm/utils/sm89_dsa_topk_triton.py`：借鉴 SGLang topk-v2 的寄存器驻留思路，整行 logits 一次 load 进片上 tile、pack(key,idx)→`tl.sort`→取 top-k（精确，仿 gemma4 routing）。是 `persistent_topk`(v1) 的替代，**待 profile 确认选择步占比后再决定是否接入** | ✅ 已写，待 profile |

**已排除**：MLA decode 换 SDPA/FlashAttention-2 —— MLA head_dim=512/576 超过 FA2 的 256 上限，内核吃不下，故 vLLM 才需专门的 FlashMLA。此路不通。

**待 re-profile**：196 ms/token 的 IPC 瓶颈是在 5.1 tok/s 配置下测的；7.16 tok/s 时驻留权重更多，瓶颈画像可能已变，需重新 py-spy 后再定后续方向。

---

## Qwen3.8-Flash-Next

**状态**: 基础适配，上游未合并  
**日期**: 2026-09-01

架构适配，确保模型可在 vLLM 中正确加载和推理。性能数据待补充。

---

*新增适配请在此文件追加同格式小节。*
