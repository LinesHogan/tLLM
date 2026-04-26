# FAQ

## 环境与安装

### FlashInfer 编译失败怎么办？

FlashInfer 是 vLLM 的可选加速后端。如果编译失败，可以先跳过它来确认基础功能正常：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

常见编译失败原因包括 CUDA toolkit 版本与 PyTorch 不匹配、GCC 版本过低等。建议直接使用 `pip install flashinfer` 安装预编译包，避免从源码编译。

### 需要什么版本的 vLLM？

tLLM 要求 **vLLM >= 0.7.2**，且仅支持 **v1 引擎**。

当前 repo-local `.venv` 的实际开发环境是 **`vllm==0.10.1.1`**，所以日常验证应优先以 `0.10.x` 行为为准。tLLM 的核心机制是在 vLLM v1 的 `GPUModelRunner` 生命周期里安装 runtime hook；v0 引擎使用完全不同的 runner 架构，无法兼容。

### `VLLM_USE_V1` 等环境变量是什么意思？

| 环境变量 | 含义 | 是否自动设置 |
|----------|------|-------------|
| `VLLM_USE_V1=1` | 启用 vLLM v1 引擎 | 是，tLLM 运行时入口自动设置 |
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | 关闭 vLLM 多进程模式 | 是，自动设置 |
| `VLLM_USE_FLASHINFER_SAMPLER=1` | 启用 FlashInfer sampler | 否，需手动设置 |
| `VLLM_DISABLE_COMPILE_CACHE` | 禁用编译缓存 | 见 [安装指南](getting-started/installation.md) |

**为什么要关闭多进程？** tLLM 的 Producer/Consumer 通过进程内全局 `STATE` 对象在 `_prepare_inputs` 和 `execute_model` 之间传递每步的定位信息。多进程模式下 worker 在独立进程中运行，全局状态无法共享。

详见 [安装指南](getting-started/installation.md)。

---

## 运行与调试

### MSE 验证失败了怎么办？

MSE 验证失败说明 parallel 模式和 gold 模式下的 hidden states 存在数值差异。排查步骤：

1. **确认环境一致**: 检查 `dtype`、`max-model-len`、`gpu-memory-utilization` 等参数是否与预期一致
2. **采样错误**: 在某些 prompt 下，next token distribution 的 topk token logprob 非常平滑，并行和顺序执行的微小差异可能导致 greedy sampling 不一致。如果你注意到 mse 在某个 token 处大涨，请观察其 token id 是否一致。如果是采样错误的原因，建议：
   - 使用确定性较强的提示词作为 producer 验证，例如数学题
   - 使用 prefill 测试避免采样错误问题
3. **放宽阈值试探**: 将 `--mse-tol` 从 `1e-4` 放宽到 `1e-3`，如果通过则说明是数值精度问题而非逻辑错误
4. **跑单元测试**: `python -m pytest -q test/test_decode_localization_unit.py` 验证定位逻辑本身是否正确
5. **逐步缩小范围**: 先跑 decode MSE，再跑 prefill MSE，定位问题在哪个阶段

详见 [调试指南](../developer-guides/debugging.md)。

### OOM 怎么排查？

常见 OOM 场景和应对：

- **模型加载阶段 OOM**: 降低 `--gpu-memory-utilization`（默认 0.8，可降至 0.5~0.6）
- **高 `sampling_n` 时 OOM**: `n=16` 等高采样场景下，sampling/sorting 路径的显存压力可能先于 ESamp 训练机制达到瓶颈。增大 `--max-model-len` (建议 `>=512`) 或降低 `n`
- **ESamp 训练阶段 OOM**: 降低 `--distiller-hidden-dim` 或 `--model-bank-rank`

### CUDA Graph capture 相关的错误？

常见原因：

1. **把不 graph-safe 的逻辑放进 hook**: 例如在 layer forward hook 里启动复杂 Python 调度、创建 stream/event、或者做可能触发同步的操作。推荐做法是 hook 里只捕获和 staging，真正的 distiller precompute 放在 `compute_logits` 边界，训练分发放在 `execute_model.post` / `out_of_band` 路径
2. **`VLLM_DISABLE_COMPILE_CACHE=1` 导致 `FileNotFoundError`**: 在 vLLM 0.7.2 下这是已知问题，执行 `unset VLLM_DISABLE_COMPILE_CACHE`
3. **`sampling_n` 过高触发 sampler CUDA assert**: 增大 `--max-model-len`

调试时可以临时使用 eager 路径来缩小问题；但生产/benchmark 路径不应默认假设必须禁用 vLLM CUDA graph。当前 ESamp 对齐吞吐实验通常会开启 ESamp 训练机制自己的 `--model-bank-train-cudagraph`，并尽量让主推理路径保留 vLLM 的 graph/compile 优化。

---

## 架构与设计

### `tllm/workflows/` 和 `tllm/verification/` 有什么区别？

当前定位是：

- `tllm.workflows.*` — 手动运行的 benchmark / repro / experiment 入口
- `tllm.verification.*` — correctness / throughput 回归验证 harness
- `test/` — pytest 断言型测试

新代码不应再把自动化验证矩阵挂在 `workflows` 下面。需要 GPU、带明确 pass/fail 语义、但又不适合塞进 pytest 的验证脚本，应该优先放进 `tllm/verification/`。

### 为什么需要 monkey-patch vLLM？

tLLM 需要在推理流水线的特定位置插入 hook 来捕获 hidden states 并触发 side-stream 训练。vLLM 没有提供足够细的原生 hook API，因此 tLLM 会在几个关键边界安装 patch。最核心的是 `GPUModelRunner` 的三个生命周期方法：

- `load_model`: 在模型加载后注册 tap layer hook
- `_prepare_inputs`: 在每步开始时快照定位信息（哪些行是 decode、哪些是 prefill）
- `execute_model`: 在推理完成后触发 runtime 内部分发、bundle 组装与异步 consumer 调度

此外，tLLM 还会安装两类更细的 hook：layer forward hook 用于捕获 hidden rows；`compute_logits` 与 sampler patch 用于采样干预。也就是说，“三个方法”是主生命周期入口，但不是全部 hook 点。

### `ConsumerFlow` 和 port 是什么关系？

`ConsumerFlow` 是 consumer 对外声明自身需求的方式：

- consumer 通过 `flows()` 声明自己读取哪些 runtime ports、写回哪些 ports、以及在哪个 window 生效
- raw runtime 事件（如 `layer.post`、`execute_model.post`）是 runtime 内部实现细节，consumer 不需要关心

这意味着：

- 如果你要写新的 consumer，优先从 `flows()` 和 `PortBundle` 开始
- 不需要理解 hook 时机或 buffer 生命周期

### DummyConsumer 是做什么的？

DummyConsumer 是一个 async read/export hidden demo，不是生产 consumer。它通过 `ConsumerFlow` 读取 `residual_stream + request_meta`，把 hidden rows 非阻塞地拷到 CPU，打印聚合摘要，并在 CPU staged copy 上注入很小的高斯噪声。

如果你要写新 consumer，可以先看 [写你的第一个 Consumer](../getting-started/write-your-first-consumer.md)。如果你想知道如何验证正确性和兼容性，去 [正确性验证](../developer-guides/validation.md)。

---

## ESamp 训练机制

### `enforce-eager` 是什么？ESamp 训练一定要用它吗？

`--enforce-eager` 是传给 vLLM 的参数，效果是禁用 CUDA graph 和 torch.compile，让所有算子以 eager 模式执行。

它对调试很有用：当你怀疑问题来自 CUDA graph capture 或 torch.compile 时，先用 eager 路径跑通，可以快速判断是算法逻辑错了，还是 graph/compile 交互错了。

但 ESamp 训练不应该天然依赖 eager。更推荐的生产路径是：hook 里只做轻量捕获和 staging，把 distiller precompute 放到 `compute_logits` 边界，把训练放到 `out_of_band` side stream；在这个基础上尽量保留 vLLM 主推理的 graph/compile 优化。

注意：`--enforce-eager` 控制的是 vLLM 主推理流水线。`--model-bank-train-cudagraph` 是 tLLM 中 ESamp 训练机制自己的训练 graph，两者独立。

### model-bank 模式何时使用？

model-bank 是专属于 ESamp training engine 的 consumer 选项。在这个引擎里，每个 prompt 可以拥有自己的小模型；如果在生成过程中为每一个模型单独发起训练 kernel，会引入明显的 launch 开销。model-bank 通过把多个训练任务合并到共享 slot / batch 中来降低这部分开销。

model-bank 模式适合以下场景：

- **高并发 per-request 训练**: 多个请求共享一组参数 slot，通过 grouped training 降低 kernel launch 次数
- **需要 CUDA graph 加速训练**: model-bank 支持 `--model-bank-train-cudagraph`，对训练 forward/backward 捕获 graph 复用
- **请求数动态变化**: model-bank 管理固定数量的 slot (`--model-bank-slots`)，请求进出时自动分配和回收

对比其他模式：
- 如果只需快速验证，用 `single` 模式即可
- 仅在某些 debug 需求中，如果请求数固定且不多，`per-request` 模式更简单
- 如果需要兼顾吞吐，`model-bank` 是推荐选择

详见 [案例：ESamp 的 Consumer 设计](../developer-guides/esamp-design.md)。
