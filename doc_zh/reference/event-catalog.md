# Runtime 事件目录

> **注意**：本页描述的是 runtime 内部的 hook 触发顺序和事件语义，用于理解 runtime 实现和调试。Consumer 作者不需要围绕这些事件名建模自己的逻辑，请优先使用 `ConsumerFlow` 和正式 port。

## 事件总览

当前 runtime 在以下时机触发内部事件：

- `load_model.pre`
- `load_model.post`
- `prepare_inputs.pre`
- `prepare_inputs.post`
- `layer.pre`
- `layer.post`
- `block.end`
- `stack.end`
- `execute_model.pre`
- `execute_model.post`

其中 `layer.post` 会携带 `HiddenBatch`，其余事件通常不带 payload。

| 事件 | 触发时机 | 默认 phase | 是否带 `HiddenBatch` | 典型用途 |
|------|----------|------------|----------------------|----------|
| `load_model.pre` | `GPUModelRunner.load_model` 前 | 无 | 否 | 资源预留、初始化前检查 |
| `load_model.post` | `GPUModelRunner.load_model` 后 | 无 | 否 | 延迟初始化、读取模型元信息 |
| `prepare_inputs.pre` | `_prepare_inputs` 前 | `decode` | 否 | step 计数、预清理 |
| `prepare_inputs.post` | `_prepare_inputs` 后，decode localization 完成后 | `decode` | 否 | 读取本 step row 定位结果 |
| `layer.pre` | tap layer forward 后、decode gather 前 | `decode` | 否 | profiling、预触发逻辑 |
| `layer.post` | tap layer decode hidden 就绪后 | `decode` | 是 | 主 hidden 消费入口 |
| `block.end` | `layer.post` 后立即触发 | `decode` | 否 | flush、checkpoint |
| `stack.end` | 当前 tap layer 是 target layer 时 | `decode` | 否 | 端到端聚合、栈边界逻辑 |
| `execute_model.pre` | `GPUModelRunner.execute_model` 前 | `decode` | 否 | step 开始协调 |
| `execute_model.post` | `execute_model` 后 | `decode` | 否 | 异步 drain、feedback 应用 |

说明：

- 当前 event dispatch 主路径是 decode-focused
- prefill hidden export 目前主要走 capture/export 路径，而不是 `HiddenBatch` event 路径

## 每个事件的语义

### `load_model.pre`

触发位置：runtime 包装 `GPUModelRunner.load_model(...)` 之前。

特点：

- 通常没有 layer path
- 不带 `HiddenBatch`

适合：

- consumer 做一次性配置检查
- 预先准备非模型依赖的资源

### `load_model.post`

触发位置：runtime 完成 hook 安装和资源分配之后。

适合：

- 读取模型 hidden size、device、layer path
- 完成依赖模型结构的懒初始化

### `prepare_inputs.pre`

触发位置：包装 `_prepare_inputs(...)` 之前。

适合：

- step 计数
- 清理上一步遗留状态

### `prepare_inputs.post`

触发位置：`_prepare_inputs(...)` 返回后，且 runtime 已经完成 decode localization 之后。

适合：

- 读取本 step 是否存在 active decode rows
- 对齐自己的 step bookkeeping

注意：

- 虽然 phase 记为 `decode`，但这里还没有 `HiddenBatch`
- 真正的 hidden rows 还要等到 tap layer forward

### `layer.pre`

触发位置：tap layer forward 结束并拿到输出 tensor 后，但 decode rows 还未 gather 成批次之前。

适合：

- 打 profiling 标记
- 准备依赖 layer 边界、但不依赖最终 batch 的逻辑

### `layer.post`

触发位置：runtime 用 `decode_row_idx` 和 `decode_valid_mask` 从 tap layer 输出中构造出 `HiddenBatch` 之后。

这是 hidden 消费的主入口。

`HiddenBatch` 当前包含：

- `step_id`
- `phase="decode"`
- `layer_path`
- `rows_hidden`
- `row_idx`
- `valid_mask`
- `prompt_idx`
- `sample_idx`
- `metadata`

适合：

- 主 hidden 消费逻辑
- 异步 GPU->CPU 传输
- ESamp distiller forward 或中间统计

### `block.end`

触发位置：每次 `layer.post` 之后立即触发。

它表示"这一层的 hidden 消费已完成"的边界信号。

适合：

- 小粒度 flush
- 分段 checkpoint

### `stack.end`

触发位置：当前 layer path 等于 runtime 配置的 target layer path 时。

它不是所有层都会触发，而是"到达当前 stack 目标边界"时才触发。

适合：

- 依赖 source/target 成对出现的聚合逻辑
- 端到端 step 内汇总

### `execute_model.pre`

触发位置：包装 `GPUModelRunner.execute_model(...)` 之前。

适合：

- 每 step 开始前的协调逻辑
- 为异步 worker 做前置状态更新

### `execute_model.post`

触发位置：`execute_model(...)` 返回后。

它是 step 末尾的关键事件，runtime 在这个事件上会调用：

```python
consumer.on_step_end(ctx)
```

适合：

- drain async queue
- synchronize side stream
- 应用 feedback
- 在 delayed backward 之后做统计

## Dispatch 时会调用哪些方法

当一个 consumer 被选中时，runtime 按下面顺序调用：

1. 如果该事件构造出了 `HiddenBatch`，调用 `consume(batch, ctx)`
2. 调用 `on_tick(event_name, ctx)`
3. 如果事件是 `execute_model.post`，再调用 `on_step_end(ctx)`

因此：

- `consume()` 不是每个事件都会触发
- `on_tick()` 是处理无 payload 生命周期事件的主入口
- `on_step_end()` 当前只在 `execute_model.post` 自动触发

## 相关文档

- [架构详解](../developer-guides/architecture.md)
- [术语表](glossary.md)
- [代码结构](project-structure.md)
- [写你的第一个 Consumer](../getting-started/write-your-first-consumer.md)
