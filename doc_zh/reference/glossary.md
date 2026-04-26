# 术语表

本文档是速查表，不是教程。如果你是第一次读，建议先看 [架构详解](../developer-guides/architecture.md)。

---

## Consumer

Hidden state 的使用方。负责接收 `PortBundle`，执行分析、训练或反馈逻辑。

核心接口（`tllm/consumers/base.py`）：
- `consumer_id` — 唯一标识
- `flows()` — 声明读写哪些 port、在哪个窗口执行
- `consume_bundle(bundle, ctx)` — 处理 runtime 组装好的数据包
- `synchronize()` — 可选的异步 flush 钩子

---

## ConsumerFlow

Consumer 对外声明自身需求的方式。一个 Consumer 可以声明多个 flow，每个 flow 说明：

- **reads**：需要读取哪些 port（如 `residual_stream`、`request_meta`）
- **writes**：需要写入哪些 port（如 `cpu_export`）
- **window**：执行时机（`background`、`same_step`、`next_step`、`out_of_band`）
- **bundle_key**：runtime 如何聚合数据帧为一个完整 bundle

定义见 `tllm/ports/base.py`。

---

## Decode localization

从 vLLM 打包后的 hidden tensor 中，找出当前 step 里所有 decode request 对应行的过程。

vLLM 把多个请求的 token 打包成一个 dense tensor。Decode localization 根据 `logits_indices` 和 `is_decode_req` 筛选出 decode phase 的行索引，写入固定 GPU buffer 供 capture layer 使用。

实现见 `tllm/producer/decode.py`。

---

## HiddenBatch

Producer 发给 Consumer 的最小数据单元，定义在 `tllm/contracts/hidden_batch.py`。

关键字段：
- `step_id` — 当前 step 编号
- `phase` — `decode` 或 `prefill`
- `layer_path` — 捕获层的模块路径
- `rows_hidden` — hidden 数据本身（**这是引用视图，不是拷贝**）
- `row_idx` — 这些行在 packed tensor 中的位置
- `valid_mask` — 哪些行有效
- `prompt_idx`、`sample_idx` — 对应哪个 prompt 和 sample

---

## Localization

从 vLLM 的 packed hidden rows 中还原出"这些行属于哪个 prompt / sample / phase"的统称。包括 decode localization 和 prefill localization 两种。

---

## Model bank

Esamp consumer 的一种参数组织方式。多个 prompt 或 request 共享一组 bank slot，而不是每个 request 独占完整模型参数。降低 kernel launch 次数，支持 CUDA Graph 加速训练。

相关实现见 `tllm/consumers/esamp/engine.py`。

---

## Phase

当前 hidden 或事件处于的生成阶段：
- `decode` — token-by-token 生成阶段
- `prefill` — 首次前向计算阶段

---

## Port

Runtime 对外暴露的正式数据接口。Consumer 通过 port 读写数据，而不是直接操作 vLLM 内部对象。

当前定义的 port 类型（`tllm/ports/base.py`）：

| Port | 含义 | 方向 |
|------|------|------|
| `residual_stream` | 各层的 residual hidden state | 可读可写 |
| `request_meta` | 请求元信息（request_id / prompt_idx / sample_idx） | 只读 |
| `cpu_export` | 异步导出到 CPU | 只写 |
| `logits` | 采样前的 logits | 只读 |
| `kv_cache` | KV cache 状态 | 可读可写 |
| `token_target` | 训练目标 token | 只读 |

每个 port 有自己的 locator 类型，用于精确定位数据。例如 `ResidualLocator(layer=0, site="block_output", phase="decode")`。

---

## PortBundle

Runtime 组装好后交给 Consumer 的数据包，定义在 `tllm/contracts/port_bundle.py`。

- `key` — 标识信息（`engine_step_id`、`phase`、`request_id`、`sample_idx`）
- `entries` — 该 ConsumerFlow 所需的所有 port 数据

Consumer 的 `consume_bundle(bundle, ctx)` 方法接收的就是这个对象。

---

## Prefill localization

从当前 packed step 中计算每个 request 的 prefill row 范围，并提取对应 hidden rows 的过程。

与 decode 不同，prefill 中每个 request 可能对应一段连续的多行。实现见 `tllm/producer/prefill.py`。

---

## Producer

负责在正确的 runtime 时机定位并导出 hidden rows 的组件。当前主要包括 decode producer 与 prefill producer。

Producer 只负责"找到并提取正确的行"，不关心这些行被用来做什么。

---

## Prompt index / Sample index

- `prompt_idx`：框架内部用于标识输入 prompt 的整数索引
- `sample_idx`：并行采样场景下，某个请求属于该 prompt 的第几个 sample

tLLM 从 vLLM 的 request id 中自动解析这两个值，支持 `sampling_n > 1`。

---

## Request id

vLLM runtime 中的请求标识字符串。对于并行采样（n>1），子请求的 id 格式为 `{sample_idx}_{parent_req_id}`。

---

## Residual stream

模型各层之间的 residual hidden state 数据流。通过 `residual_stream` port 暴露给 Consumer。

支持的 site（位置）：`block_input`、`attn_input`、`attn_output`、`mlp_input`、`block_output`。

Consumer 可以通过 `ResidualStream.read(layer=0, site="block_output", phase="decode")` 精确指定需要哪一层、哪个位置的数据。

---

## RuntimeContext

发给 Consumer 的运行时上下文，定义在 `tllm/contracts/runtime_context.py`。

包含 `runner`、`model`、`device`、`main_stream`、`is_compiling`、`uses_cudagraph`、`event_name`。

Consumer 通常只需要把它当作辅助上下文，不需要深入理解每个字段。

---

## Runtime adaptation

在推理过程中使用捕获到的 runtime 数据更新辅助状态的模式，通常放在 side stream 上和主推理流水线重叠执行。ESamp 的 distiller 训练机制是其中一个例子。

---

## Source layer / Target layer

ESamp 中的一对概念：
- **Source layer**：产生输入 hidden 的层（默认第一层）
- **Target layer**：产生监督目标 hidden 的层（默认最后一层）

---

## Tap layer

Runtime 实际插入 forward hook 的层。默认配置通常是 `model.model.layers[0]`，也可以通过 layer index 或显式 path 指定。

---

## Window

ConsumerFlow 的执行时机声明：
- `background` — 异步后台执行，不阻塞主推理
- `same_step` — 当前 step 内同步完成
- `next_step` — 结果在下一步生效
- `out_of_band` — step 末尾异步 side-stream work

---

## 相关文档

- [架构详解](../developer-guides/architecture.md)
- [Port 目录](port-catalog.md)
- [写你的第一个 Consumer](../getting-started/write-your-first-consumer.md)
