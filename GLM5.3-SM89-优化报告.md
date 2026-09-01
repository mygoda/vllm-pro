# GLM-5.3-Flash 在 8×RTX 4090 (sm_89) vLLM 部署 —— 完整技术报告

**日期**: 2026-09-01
**结论**: GLM-5.3-Flash（320B-A18B）在纯消费级 8×4090 上,通过 vLLM 端到端**输出正确**、支持长上下文,并经 CUDA Graph 提速至 **5.1 tok/s**。此前 vLLM 与 SGLang 在此硬件上均只能输出乱码,业界无成功先例。

---

## 1. 硬件与软件环境

| 项 | 规格 |
|---|---|
| GPU | 8× NVIDIA RTX 4090（sm_89 / Ada），24.5 GB/卡，共 196 GB VRAM |
| 内存 | 1 TB DDR |
| CUDA | 13.0 |
| 引擎 | vLLM 0.28.1 + GLM-5.3 PR（源码 `/root/vllm-glm` glm53 分支；venv `/root/vllmglm`） |
| 模型 | GLM-5.3-Flash：320B 总参 / 18B 激活 MoE，288 专家选 8，native FP8，306 GB |

**模型架构要点**（决定了移植难度）:
- 45 层 = 34 层 **KDA 线性注意力**（gated-delta-net）+ 11 层 **DSA 稀疏 MLA**（deepseek_sparse_attention）
- **mHC**（multi-head hyper-connection）：维护 hc_mult=4 条并行残差流,层间混合,RMSNorm 融合进 mHC kernel
- **NoPE**：qk_rope_head_dim=0（无旋转位置编码），qk_nope_head_dim=256，kv_lora_rank=512
- DSA indexer：index_kpool=4，index_n_heads=32，index_topk=2048
- 多模态外壳（`Glm5NextForConditionalGeneration`），本次仅用文本路径

---

## 2. sm_89 的硬约束（为何默认跑不起来）

RTX 4090（Ada / sm_89）相对 Hopper（sm_90）缺失：TMA、wgmma、thread-block clusters；FP4 张量核（Blackwell）；每 block 共享内存上限低。直接后果:

- **DeepGEMM 全部内核仅支持 Hopper/Blackwell**（`is_deep_gemm_supported()` 在 sm_89 返回 False）。GLM 的 DSA indexer logits（`fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits`）**只有 deepgemm 一条路,无任何 fallback**。
- vLLM 所有 sparse-MLA 后端（FLASHMLA_SPARSE、FLASHINFER_MLA_SPARSE_SM90/120…）**全 gate 在 Hopper/Blackwell**。sm_89 上起服务直接报 `No valid attention backend (head=512, use_mla, use_sparse)`。
- tilelang（mHC 用）的 JIT 在 sm_89 上与 apache-tvm-ffi 0.1.13 冲突,C++ 层 `type index 132 already registered` 静默 terminate 崩 worker。

---

## 3. 核心难题：输出乱码，且查了几十轮

打通管线后模型能生成 token,但**输出是乱码**（如 `=Eajan_attributes 以前以前`）。特征极具迷惑性:

- **保幅度、语义错**：不是 NaN、不是爆炸。逐层 hook 显示 45 层残差范数稳定在 115~137,全程无 NaN。
- **确定性**：同一 prompt 三次输出完全一致（排除随机未初始化状态）。
- **精度无关**：bf16 与 fp8 产出**完全相同的乱码**（排除量化精度）。
- **top-token logprob = -5**（正常模型应 > -1；均匀分布 = -11.9）——模型"半懂非懂"。

### 逐一排除（全部可复现）

| 组件 | 排除依据 |
|---|---|
| fp8 量化/精度 | bf16 同样乱码 |
| MoE 内核 | marlin 与 triton 结果一致 |
| DSA kpool logits 内核 | 短 prompt 下 `short_prefill` 短路,内核根本没被调用 |
| chunk-KDA 内核 | 自洽测试:整段 vs 分两半链 state,REL=0.0 |
| 自写 MLA kernel 数学 | 对拍独立稠密参考 REL=0.0014（纯 bf16 舍入） |
| Tokenizer | vLLM 与 transformers 输出相同 token ids |
| KV cache 写入顺序 | `do_kv_cache_update` 在 `forward_impl` 之前,顺序正确 |
| 权重加载 | embedding shape/值合理,无 loading error |
| chat template | completions 与 chat/completions 均乱码 |

**每个组件单独测都对,组装起来却错** —— 这指向"接线"（composition），而非任何单个 kernel。

### 真凶：native mHC 路径漏了融合 RMSNorm

读 `Glm5NextDecoderLayer.forward` 梳理数据流时发现:每层调用
```python
self.hc_pre(x, ..., norm_weight=self.input_layernorm.weight.data)
```
即 **input_layernorm / post_attention_layernorm 的 RMSNorm 被融合进 mHC pre kernel**。tilelang 版 `mhc_pre_tilelang` 支持 `norm_weight` 并应用 RMSNorm；但 **native 版 `mhc_pre_torch` 根本没有 `norm_weight` 参数,`MHCPreOp.forward_native` 接收后静默丢弃**。

我为绕开 tvm_ffi 崩溃把 `has_tilelang()` 强制 False → 强制走 native → **每层 layernorm 的学习权重全被丢** → attention/MLP 收到幅度差 **13×** 的未归一化输入 → 乱码。而残差范数因 mHC 混合系数是 sigmoid 有界,看着"健康",所以一直没暴露。

**冒烟证据**:tilelang(带 norm) vs torch(不带) 的 layer_input，REL=11.6，范数 11.5 vs 144。

**关键教训**:我最初做 mHC 对拍时**两边都没传 norm_weight**,所以早期误判 tilelang≡native、以为 mHC 无关。这是漏了一个"隐性入参"导致的方向性误判。

---

## 4. 让它跑对 —— mHC norm 修复（3 处，对拍 REL=0.0028）

1. `kernels/mhc/torch.py::mhc_pre_torch`：新增 `norm_weight` / `norm_eps` 参数,对 layer_input 施加标准 RMSNorm：`x · rsqrt(mean(x²)+eps) · weight`（fp32 累加，复现 tilelang big_fuse 的 norm-fused write path）。
2. `layers/mhc.py::MHCPreOp.forward_native`：把 `norm_weight, norm_eps` 传给 mhc_pre_torch（原来丢弃）。
3. `layers/mhc.py::MHCFusedPostPreOp.forward_cuda`：post→pre 分解同样透传 norm_weight。

修复后 top-token logprob 从 **-5 → -0.5**。

---

## 5. 让它跑快 —— 自写 sm_89 内核 + 解锁 CUDA Graph

### 5.1 自写 sm_89 sparse-MLA 后端
`vllm/v1/attention/backends/mla/torchref_mla_sparse_sm89.py`：gather topk KV + torch einsum，NoPE-aware，分块省显存。数学对拍独立稠密参考 REL=0.0014。注册进 registry.py + cuda.py 选择器（head_size==512）。承担全部 11 层 DSA-MLA 注意力（deepgemm/flashmla 全 Hopper-gate）。

### 5.2 自写 sm_89 长上下文 DSA logits 内核
`vllm/utils/deep_gemm.py`：
- `_sm89_torch_ref_mqa_logits`（prefill）+ `_sm89_torch_ref_paged_mqa_logits`（paged decode）
- 数学：`logits[m,n] = Σ_h relu(q[m,h]·k[n]) · weights[m,h]`，因果 + 上下文 mask
- paged 版解码 fp8 paged cache `[nb, bs, 1, D+4]`（每 (block,pos)：D 个 fp8-e4m3 值字节 + 4 字节 fp32 scale）
- **关键：paged 版写成完全向量化、纯 GPU、无 `.item()`/`.tolist()`/python 循环** → **CUDA-graph-capturable**。第一版用循环+`.item()`，capture 时报 `Cannot copy between CPU and CUDA tensors during CUDA graph capture`；重写为全向量化后通过。
- `!is_deep_gemm_supported()` 时三个函数（mqa / paged_mqa / get_paged_meta）自动路由到 torch fallback。

### 5.3 解锁 CUDA Graph（这是提速的关键）
问题链条:去掉 `--enforce-eager` 想用 CUDA Graph → capture 用长上下文预热 → 超过 topk=2048 → 触发未 guard 的 deepgemm `fp8_fp4_paged_mqa_logits` → 撞 `attention.hpp:320` 的 `arch 9/10/12` 断言 → 崩。

解法:上面 5.2 的向量化 torch DSA fallback 让 capture 不再触发 deepgemm。再配合:
- `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'`（单用户只需小 batch，省 capture 显存）
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（避免 capture OOM）

---

## 6. 测试结果

### 6.1 正确性（全部通过）

| 输入 | 输出 |
|---|---|
| 中国的首都是 | 北京，1400 多年前，隋朝工匠… |
| 1 2 3 4 5 | 6 7 8 9 10 |
| 水的化学式是 | H2O，O2 是氧气… |
| The capital of France is | Paris. |
| def add(a, b): | return a + b |
| chat: 解释机器学习 | 连贯推理输出 |
| **长上下文（>2048 token 输入）** | **连贯**（验证 paged DSA fallback 字节布局正确） |

### 6.2 速度

| 配置 | decode | TTFT | 提升 |
|---|---|---|---|
| eager（初版，enforce-eager 强制） | 3.5 tok/s | 0.50 s | 基线 |
| **CUDA Graph（当前）** | **5.1 tok/s** | **0.37 s** | **1.46×** |
| 长 prompt（>2048）prefill | — | ~18 s | torch DSA fallback 慢（正确性优先） |

decode 速度三次重复测 5.09 / 5.09 / 5.09 tok/s，稳定。

### 6.3 显存分解（每卡，24.5 GB）

| 项 | 占用 |
|---|---|
| 驻留权重 + non-torch | 0.71 GiB（cpu-offload 40，权重几乎全在 CPU 流式） |
| 峰值激活 | 1.89 GiB |
| CUDA Graph 池 | 0.39 GiB |
| **KV cache** | **15.74 GiB** |

---

## 7. 瓶颈分析（为何是 5 tok/s，而非更快）

- **eager 阶段**：GPU util 100% 但功耗仅 91 W（TDP 450 W 的 20%）、显存带宽 0% → **逐 op launch 开销**（既非计算、也非带宽、也非 PCIe；实测 cpu-offload 28 vs 36 对速度无影响）。
- **CUDA Graph 阶段**：py-spy 显示 8 个 worker 全在 `shm_broadcast.acquire_read` 等待 → 瓶颈转移到 **TP=8 的驱动/IPC 协调开销**（~196 ms/token）。GPU 计算已被 graph 化,剩下的是 CPU 侧多进程同步。
- **根本约束**：320B 模型（306 GB） > 196 GB VRAM，cpu-offload 强制；sm_89 无 Hopper 内核,部分路径强制 eager。

要再快需深改 vLLM 内核（让 KDA/MLA 也 capturable、或降低 IPC 开销），属另一量级的工作。

---

## 8. 已知限制 & 可优化项

1. **长 prompt prefill 慢**（~18 s）：走我的 torch DSA fallback（全窗口 einsum）。可写 triton 版提速。
2. **KV cache 偏大**（15.74 GiB/卡）：当前为多并发预留。单用户可用 `--kv-cache-memory` 砍到 3~4 GB，把显存让给驻留权重（降 cpu-offload），可能进一步降低流式开销——待验证。
3. **KDA / 自写 MLA kernel 是 eager_break 片段**：若改造成 capturable，CUDA Graph 覆盖更全,decode 可更快。
4. **仅验证到 8192 上下文**：更长需评估 torch DSA fallback 的显存/速度。

---

## 9. 复现指南

改动 10 个文件（`git diff --stat`，共 +189 行）,核心:
- `kernels/mhc/torch.py` + `layers/mhc.py`：mHC 融合 RMSNorm 修复（**跑对的关键**）
- `backends/mla/torchref_mla_sparse_sm89.py` + `registry.py` + `platforms/cuda.py`：自写 sm_89 sparse-MLA 后端
- `utils/deep_gemm.py`：自写 sm_89 DSA logits 内核（prefill + 向量化 paged）+ 自动路由
- `utils/import_utils.py`：`has_tilelang()`→False
- `backends/mla/indexer.py`：`has_deep_gemm()`→`is_deep_gemm_supported()`
- 依赖:**apache-tvm-ffi 降 0.1.11**（0.1.13 会静默崩 worker）

启动:`/root/run_vllm_glm.sh`（tp8, cpu-offload-gb40, gpu-util0.78, max-len8192, moe-backend marlin, cudagraph[1,2,4,8], expandable_segments, port 8095）。

---

## 10. 一句话总结

> 在业界"vLLM/SGLang 都跑不出正确 GLM-5.3 on 4090"的普遍结论下，定位到真凶是 **native mHC 路径漏实现了融合 RMSNorm**（官方假设 CUDA 上永远有 tilelang），修复后再自写 sm_89 的 sparse-MLA 后端与长上下文 DSA logits 内核（向量化以支持 CUDA Graph），最终在 8×4090 上端到端正确输出、支持长上下文、5.1 tok/s 单用户可用。

## 附:调优结果(2026-09-01 补充)

CUDA Graph 消除 launch 开销后,py-spy 显示瓶颈转为权重流式。**把单用户用不到的 KV cache 换成驻留权重**:
| 配置 | decode | KV容量 |
|---|---|---|
| cudagraph, cpu-offload40, KV15.7GB | 5.1 tok/s | 845K token(103×并发) |
| **cudagraph, cpu-offload24, KV4GB(214K token/26×)** | **7.16 tok/s** | 最优,已锁定 |
| cpu-offload18 | OOM | 显存到顶 |

**最终 7.16 tok/s = 最初 eager(3.5) 的 2.05×。** 显存已到 4090 天花板(offload18 即 OOM)。
关键:降 cpu-offload(40→24)多驻留 ~14GB/卡权重 + `--kv-cache-memory 4294967296` 砍 KV。启动脚本已更新。
