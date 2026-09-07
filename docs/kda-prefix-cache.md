# GLM-5.3 KDA Prefix Caching：现状、启用、缺口

> 结论先行：**vLLM 已内置 KDA/mamba prefix caching**，GLM-5.3 不需要重建，先开 flag。
> 更进阶的 RecoverSSM 被 gate 在 Kimi 系，扩到 GLM-5.3 是一个可选的真代码贡献。

## 背景：为什么线性注意力的 prefix cache 特殊

GLM-5.3 有 34 层 KDA（gated-delta-net，mamba 家族）。普通 attention 缓存 per-token KV；KDA 没有 per-token KV，只有一个随前缀滚动的 recurrent state。默认每个请求都从头 recompute 整段前缀的 KDA state —— 系统提示 / 多轮对话是纯浪费。

## 现状：基建已在 vLLM 里

`Glm5NextLinearAttention(GatedDeltaNetAttention)` 是 mamba 家族层，自动走 `MambaManager`。vLLM 的 `mamba_cache_mode`（`vllm/config/cache.py:188`）：

| 模式 | 含义 |
|---|---|
| `none`（**默认**） | 不缓存 mamba state，每请求重算 |
| `align` | 缓存每个 scheduler step 最后 token 的 mamba state，prefix 命中时复用（= prefix caching） |
| `all` | 更激进 |

`MambaManager`（`single_type_kv_cache_manager.py:1368`）在 align 模式下已实现 checkpoint block、`find_longest_cache_hit`、fine-grained hash lookup —— 等价于 SGLang 的 `mamba_radix_cache`。

## Tier 1：零代码启用（先做这个）

```bash
--mamba-cache-mode align --enable-prefix-caching
```

约束（`vllm/config/vllm.py:2814`）：
- align 模式要求 chunked MM input（`disable_chunked_mm_input` 必须 False）—— GLM-5.3-Flash 走文本路径，默认满足
- 建议配 Triton mamba backend

**验证**：开启后压一条固定长系统提示 + 变化 query 的多轮请求，看第二轮起 KDA prefill 是否被 prefix 命中跳过（TTFT 应明显下降）。

## Tier 2：RecoverSSM 扩到 GLM-5.3（可选，仅 MTP 需要）

`--use-replayssm`（RecoverSSM）是 Kimi-K3 的进阶机制：维护中间 state 的 ring buffer，让 **spec-decode verify** 能从任意接受点恢复 state（对应 SGLang #33722 fused-accept）。当前 gate 死在 Kimi（`vllm/config/vllm.py:2870`）：

```python
if architecture not in ("KimiLinearForCausalLM", "KimiK3ForConditionalGeneration"):
    raise ValueError("RecoverSSM is only supported for Kimi-K3 KDA")
```

GLM-5.3（`Glm5NextForConditionalGeneration`）被排除。扩过来需要三步：

1. `Glm5NextForConditionalGeneration` 加 `SupportsReplaySSM` mixin（`interfaces.py:1187`）
2. KDA 的 `get_state_shape` / dtype 补 replayssm ring-buffer 追加（照 `nemotron_h.py:748,785` 的 `append_replayssm_ring`）
3. 把 `Glm5NextForConditionalGeneration` 加进上面 allowlist

约束：align 模式 + RecoverSSM 需 `VLLM_USE_V2_MODEL_RUNNER=1`、`pipeline_parallel_size=1`、Triton mamba backend、非 stochastic-rounding cache。

**判据**：只有当你在 GLM-5.3 上跑 MTP 投机解码、且 profile 显示 KDA state commit scatter 是瓶颈时才值得做。底层 `fused_recurrent_kda` 已支持 `num_accepted_tokens`/`initial_state`（你 spec 路径在用），接口是通的。

## 净结论

| 项 | 判定 |
|---|---|
| 普通 KDA prefix caching | ✅ vLLM 已有，`--mamba-cache-mode align` 零代码开 |
| 不要重建 checkpoint pool | ❌ 会重复 `MambaManager` |
| RecoverSSM for GLM-5.3 | ⚠️ 真代码缺口，但仅 MTP 场景，按需做 |
