# 代码结构参考

这篇文档回答两个核心问题：

1. **这个仓库的代码按什么职责分层？**
2. **如果我想改某类行为，应该先看哪里？**

下面是按职责分层的目录速查，以及每种改动对应的入口文件。如果你只是想了解概念，不需要记住所有路径——改到具体问题时再来查。

## 顶层目录

| 目录 | 内容 |
|------|------|
| `doc/` | 用户文档 |
| `test/` | 单元测试 |
| `tllm/` | 核心实现 |
| `outputs/` | 运行产物 |

## `tllm/` 分层

### `tllm/common/` — 共享状态

- `state.py` — 全局 STATE、配置、layer 解析
- `runtime_step_state.py` — 从 `_prepare_inputs` 快照 step 元数据
- `path_resolution.py` — 解析 layer path，兼容不同模型结构

**什么时候看这里**：decode hidden 对齐不对，或者想改 capture layer 解析规则。

### `tllm/contracts/` — 数据契约

Producer 和 Consumer 之间的共享数据结构。

- `hidden_batch.py` — HiddenBatch
- `runtime_context.py` — RuntimeContext
- `port_bundle.py` — PortBundle、BundleKey

**什么时候看这里**：写新的 consumer，想确认 bundle 里有什么字段。

### `tllm/ports/` — 正式接口

Consumer 对外声明需求时使用的接口定义。

- `base.py` — PortKind、ConsumerFlow、Window
- `residual_stream.py` — ResidualStream、ResidualLocator
- `request_meta.py` — RequestMeta
- `cpu_export.py` — CpuExport

**什么时候看这里**：声明新的 ConsumerFlow，或者想理解某个 port 的 locator 语义。

### `tllm/producer/` — 核心算法

从 packed tensor 中定位并提取 hidden rows。

- `decode.py` — decode localization、graph-safe gather
- `prefill.py` — prefill localization、eager-first 选取

**什么时候看这里**：Decode MSE 不对，或者想知道 packed rows 是怎么被还原的。

### `tllm/consumers/` — 消费端

- `base.py` — 通用 consumer 接口（`flows()`、`consume_bundle()`）
- `dummy/` — 最小示例，适合学习如何写 consumer
- `esamp/` — 内置的 adaptive/guidance consumer
  - `consumer.py` — `ESampConsumer`，BaseConsumer 包装层
  - `engine.py` — `ESampTrainEngine`，训练引擎与其本地 helper
  - `model.py` — ESamp 训练机制使用的低秩 distiller 模型定义
  - `template.py` — model-bank 初始化模板与抽取逻辑

**什么时候看这里**：从 dummy 复制新 consumer，或者排查 ESamp 训练问题。

### `tllm/runtime/` — 运行时桥接

把 vLLM 和 tLLM 连接起来。

- `vllm_patch/port_runtime_hooks.py` — hook 生命周期和 runtime 编排入口
- `vllm_patch/common_hooks.py` — 构造 RuntimeContext，分发 legacy event
- `vllm_patch/sampler_patch.py` — 把 sampler state 暴露给 tLLM sampler bridge，并调度 distiller precompute
- `vllm_patch/adapters/` — 隔离不同 vLLM 版本的 `_prepare_inputs` 输出结构
- `dispatch_plan.py` — flow target 静态路由
- `ports/` — PortFrame、BundleAssembler、residual binding 等内部实现

**什么时候看这里**：consumer 没收到 bundle，或者想确认 hook 时机。

注意：这些是 runtime 内部实现，consumer 作者通常不需要深入理解。ESamp 的训练细节不在 runtime 里，而在 `tllm/consumers/esamp/engine.py`。

当前 generic runtime host 在：
- `tllm/runtime/residual_runtime.py`

### `tllm/workflows/` — 手动工作流

benchmark、repro、experiment 的入口脚本。

### `tllm/verification/` — 回归验证

GPU verification harness，带 pass/fail 语义。

## 常见改动先看哪里

### Decode hidden 不对

1. `tllm/producer/decode.py`
2. `tllm/common/runtime_step_state.py`
3. `test/test_decode_localization_unit.py`

### 想写新 consumer

1. `tllm/consumers/base.py`
2. `tllm/consumers/dummy/consumer.py`
3. `tllm/ports/base.py`
4. `tllm/ports/residual_stream.py`

### Consumer 没收到 bundle

1. 检查 `flows()` 声明
2. `tllm/runtime/dispatch_plan.py`
3. `tllm/runtime/ports/residual_bundle_dispatch.py`

## 相关文档

- [架构详解](../developer-guides/architecture.md)
- [术语表](glossary.md)
