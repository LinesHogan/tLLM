# 安装指南

这篇文档教你如何从零开始安装 tLLM。

## 1. 克隆仓库

```bash
git clone <your-repo-url>
cd tLLM
```

## 2. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

需要 Python >= 3.10。

## 3. 安装 vLLM

```bash
pip install vllm
```

tLLM 本身不需要 `pip install`，以源码方式直接运行。但它依赖 vLLM 作为底层推理引擎。

vLLM 版本要求：
- 最低 `vllm >= 0.7.2`
- **仅支持 v1 引擎**，v0 引擎使用完全不同的 runner 架构，无法兼容
- 当前 tLLM 主要开发和验证的版本是 `vllm==0.10.x`，建议优先使用这个版本

安装后验证：

```bash
python -c "import vllm; print(vllm.__version__)"
```

## 4. 安装 tLLM

```bash
pip install -e .
```

这会以 editable 模式安装 tLLM，让你可以改代码立即生效。

## 5. 验证 GPU 环境

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

如果 CUDA 不可用，检查你的 PyTorch 是否安装了 CUDA 版本。

## 6. 验证 tLLM 安装

对普通用户来说，最直接的验证方式是跑内置的 ESamp starter：

```bash
python starter.py --max-new-tokens 32
```

这个命令会加载 `Qwen/Qwen2.5-7B-Instruct`，并行生成 16 条回答，同时运行 ESamp 的训练机制。环境正常时，输出末尾会看到类似这样的统计：

```text
ESamp stats: loss_avg=... loss_count=... answers=16 ...
```

其中 `loss_count > 0` 说明 consumer 确实拿到了 hidden state，并触发了 ESamp 的训练机制。

如果遇到 `ModuleNotFoundError`，先检查虚拟环境是否激活；如果遇到 OOM，可以降低 `--gpu-memory-utilization`，或者临时把模型换成更小的 `Qwen/Qwen2.5-0.5B-Instruct`。

开发者如果要验证 hidden localization 的数值正确性，可以再跑 `verify_v1_decode_rows_minimal.py` 这类 MSE correctness check。它属于框架开发验证，不是日常安装流程的必经步骤。

## 环境变量

以下变量由 tLLM 运行时入口**自动设置**，不需要手动配置：

| 变量 | 值 | 作用 |
|------|-----|------|
| `VLLM_USE_V1` | `1` | 强制启用 v1 引擎 |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` | 关闭多进程。因为 Producer/Consumer 通过进程内全局状态传递每步定位信息，多进程下状态无法共享 |

建议手动开启的：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=1
```

FlashInfer sampler 对推理吞吐有明显提升，建议在吞吐实验中统一开启。如果编译失败，可以先跳过：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

## 常见问题

**FlashInfer 编译失败怎么办？**

FlashInfer 是 vLLM 的可选加速后端。如果编译失败，先跳过它确认基础功能正常。常见原因包括 CUDA toolkit 版本与 PyTorch 不匹配、GCC 版本过低等。建议直接用 `pip install flashinfer` 安装预编译包。

**`VLLM_DISABLE_COMPILE_CACHE=1` 导致 `FileNotFoundError`？**

在 vLLM 0.7.2 下这是已知问题，执行 `unset VLLM_DISABLE_COMPILE_CACHE`。

## 下一步

- [运行一个 Consumer](run-consumer.md) — 如果你想直接体验内置算法
- [写你的第一个 Consumer](write-your-first-consumer.md) — 如果你想开发自己的逻辑
