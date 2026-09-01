# vllm-pro

**vLLM fork，专注于上游还未支持的模型和显卡的快速适配与调优。**

上游 [vllm-project/vllm](https://github.com/vllm-project/vllm) 合并新模型/新硬件往往需要数周到数月，这个仓库的目标是把这个等待窗口压缩到最短，并在适配过程中做好内核级优化。

---

## 已支持（相对上游的增量）

| 模型 | 显卡 / sm | 状态 | 详情 |
|------|-----------|------|------|
| GLM-5.3-Flash | RTX 4090 (sm_89) | ✅ 跑通 + 内核优化 | [优化报告 →](docs/optimizations.md#glm-53-flash--rtx-4090-sm_89) |
| Qwen3.8-Flash-Next | — | ✅ 适配 | [详情 →](docs/optimizations.md#qwen38-flash-next) |

---

## 工作范围

- **模型适配**：上游还没合并的模型架构，先在这里跑起来
- **硬件适配**：上游不支持或支持不好的 GPU 计算架构（sm 版本）
- **内核调优**：decode indexer、attention kernel、GEMM 等针对具体硬件的优化
- **同步上游**：定期 rebase，适配成熟后尽量往上游推

---

## 快速开始

```bash
# 推荐用 uv
uv venv --python 3.12
source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

其他安装选项参考 [上游文档](https://docs.vllm.ai/en/latest/getting_started/installation.html)。

---

## 使用方式

与上游 vLLM 完全兼容，直接替换即可：

```bash
vllm serve <model> --...
```

---

## 贡献

适配新模型或新硬件前，先确认上游是否已有在途 PR（见 [AGENTS.md](AGENTS.md)）。
