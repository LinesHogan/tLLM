# 架构详解

这篇文档解释 tLLM 的内部架构：数据怎么从 vLLM 里流出来，经过哪些组件，最后到达 consumer。

如果你只想写 consumer，不需要读完这篇。先看 [写你的第一个 Consumer](../getting-started/write-your-first-consumer.md)，遇到不懂的概念再来查。这篇文档主要面向想理解 runtime 工作原理的开发者。

tLLM 的核心设计是把"提取数据"和"使用数据"拆开：

- **Producer** 只管从 packed tensor 中定位并提取需要的行
- **Consumer** 只管拿到这些行之后做什么
- **Runtime** 做桥接：装 hook、组装 bundle、调度 feedback

下面先解释这个设计要解决什么问题，再展开每个组件。

## 从问题出发

假设你想在 vLLM 推理过程中捕获每一层 transformer block 的 hidden state，做分析，然后可选写回影响下一层。

你会遇到三个问题。

### 问题 1：vLLM 把多个请求打包成一个 dense tensor

4 个请求同时跑，vLLM 拼成 `[total_tokens, hidden_size]`。你需要知道"第 3 行属于哪个请求、是不是 decode 阶段"。这叫 **localization**。

### 问题 2：捕获时机和 vLLM 版本强耦合

`_prepare_inputs`、`execute_model`、layer forward 在不同版本里接口会变。把代码写死在 forward 里，每次升级 vLLM 都要重写。

### 问题 3：训练逻辑和推理混在一起

在 forward 里直接启动 ESamp distiller backward，会阻塞主推理线程，吞吐崩掉。需要异步执行，但又必须和主 stream 正确同步。

---

## tLLM 的做法

把"提取数据"和"使用数据"拆开：

- **Producer** 只管一件事：在正确时机从 packed tensor 中定位并提取需要的行
- **Consumer** 只管一件事：拿到这些行之后做什么分析、训练或反馈
- **Runtime** 做桥接：装 hook、组装 bundle、调度 feedback

当前内置的 ESamp 就遵循这个边界：
- Runtime 负责 capture / localization / bundle assembly
- `ESampConsumer` 负责消费 bundle
- `ESampTrainEngine` 负责训练流水线、model-bank、shared-graph

Producer 的改动只影响"怎么找到正确的行"。Consumer 的改动只影响"拿到行之后做什么"。两者独立演进。

---

## 核心抽象：Port

Runtime 和 Consumer 之间不共享 vLLM 内部对象，而是通过正式的数据接口交互。

当前定义的 port 类型：

| Port | 含义 | 例子 |
|------|------|------|
| `residual_stream` | 某层某位置的 hidden state | `ResidualStream.read(layer=0, site="block_output")` |
| `request_meta` | Request identity 信息 | `RequestMeta.read()` |
| `cpu_export` | 异步导出到 CPU | `CpuExport.write(channel="my_analysis")` |
| `logits` | 采样前的 logits | `Logits.read()` |
| `kv_cache` | KV cache 状态 | `KVCache.read(layer=12)` |

Consumer 通过 `ConsumerFlow` 声明自己要读写哪些 port：

```python
ConsumerFlow(
    reads=(
        ResidualStream.read(layer=0, site="block_output", phase="decode"),
        RequestMeta.read(),
    ),
    writes=(CpuExport.write(channel="debug"),),
    window="background",
)
```

Runtime 负责：
- 在 consumer 声明的 layer 上装 hook
- 每步从 packed tensor 中收集这些 port 的数据
- 按 `bundle_key` 聚合成一个 `PortBundle`
- 在合适的 window 调用 `consume_bundle(bundle, ctx)`

---

## 一次 step 内发生了什么

以 decode 为例，从一个 Consumer 的视角看一次 step：

**Step 1：vLLM 准备输入**

`_prepare_inputs` 被调用。Runtime 在这里：
- 快照当前 step 的 request 顺序和 metadata
- 计算 decode localization：哪些行属于 decode phase、对应哪个 prompt

**Step 2：Layer forward**

vLLM 执行模型推理。在 consumer 声明的 layer 上，runtime 的 hook 被触发：
- 用固定 GPU buffer 做 graph-safe gather：从 packed hidden 中选出 decode 行
- 同时 stash prefill 行（如果启用了 prefill producer）

**Step 3：Bundle 组装**

`execute_model` 返回后，runtime 从所有 hook 结果中：
- 提取 consumer 需要的 port 数据
- 按 `bundle_key` 聚合
- 调用 `consumer.consume_bundle(bundle, ctx)`

**Step 4：Feedback（可选）**

如果 consumer 需要 step 末尾动作，或者实现了 `on_step_end()`：
- 在 step 末尾触发 feedback
- ESamp 在这里排队 delayed backward 到 side stream

**Step 5：Capture export（可选）**

如果启用了 capture 模式：
- 把 localized hidden rows 导出到 CPU 存储
- 用于后续的 MSE 对齐验证

---

## Localization

Producer 的核心工作，分 decode 和 prefill 两种。

### Decode localization

vLLM 的 packed tensor 中，每个请求在 decode phase 只占**一行**（当前正在生成的那个 token）。

Producer 的做法：
1. 读取 `logits_indices[i]` —— 第 i 个请求的采样行在 packed tensor 中的位置
2. 筛选出 `is_decode_req=True` 的请求
3. 把这些行索引写入固定 GPU buffer（`decode_row_idx`）
4. 在 hook 中用 `index_select(..., out=decode_hidden_rows_buffer)` gather 到固定 buffer
5. 用 `valid_mask` 标记有效行

为什么用固定 buffer？因为 CUDA Graph 回放时不能每步分配新 tensor。

### Prefill localization

Prefill 不同：每个请求可能对应**一段连续的多行**。

Producer 的做法：
1. 计算 `prefill_len = min(scheduled, prompt_len - computed)`
2. 记录每个请求在 packed tensor 中的 `[start, start + prefill_len)` 范围
3. 用动态 row index tensor 取出这些行
4. 导出到 CPU capture 存储

Prefill 目前走 eager-first 路径，与 decode 的 graph-safe 路径解耦。

---

## Consumer 怎么接入

最简单的方式：继承 `BaseConsumer`，实现三个方法。

```python
class MyConsumer(BaseConsumer):
    @property
    def consumer_id(self):
        return "my_consumer"

    def flows(self):
        return [
            ConsumerFlow(
                reads=(ResidualStream.read(layer=0, site="block_output", phase="decode"),),
                window="background",
            )
        ]

    def consume_bundle(self, bundle, ctx):
        hidden = bundle.entries["hidden"]
        # 你的处理逻辑
```

如果需要 step 末尾动作（drain 队列、启动 backward），再实现 `on_step_end(ctx)`。

完整的扩展教程见 [写你的第一个 Consumer](../getting-started/write-your-first-consumer.md)。

---

## 相关文档

- [术语表](../reference/glossary.md) — 所有核心概念速查
- [代码结构](../reference/project-structure.md) — 改代码时该看哪些文件
