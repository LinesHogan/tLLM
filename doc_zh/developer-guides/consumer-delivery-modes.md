# Consumer 投递模式

大多数 consumer 都应该使用普通的 bundle 路径。它稳定、容易测试，也让你的代码不
需要关心 runtime 内部怎么管理 buffer。

Device-lease 路径是给更窄的场景准备的：consumer 每个 decode step 都要在 GPU 上
继续工作，希望把这些工作和生成过程重叠起来，而且愿意遵守更严格的 buffer 生命周期
约定。ESamp 会用这条路径，因为它是一个带 runtime adaptation 和 sampler guidance
能力的复杂 consumer，其中训练只是它的一种机制。不要把 ESamp 理解成“训练路径”的
同义词。

## 普通 bundle 路径

默认写法是：

```python
ConsumerFlow(
    reads=(...),
    writes=(),
    window="background",
    bundle_key=("engine_step_id", "phase"),
)
```

不显式设置 `delivery` 和 `ownership` 时，含义是：

```python
delivery="bundle"
ownership="borrowed"
```

Runtime 会组装一个 `PortBundle`，每个被声明读取的 port 都直接出现在
`bundle.entries` 里。如果你的 flow 读取了一个 `role="hidden"` 的 residual
stream，就这样取：

```python
hidden = bundle.entries.get("hidden")
```

这里的 tensor 要按 borrowed view 理解：可以在 `consume_bundle()` 里读取；如果要
跨 step 保存，就自己 clone。这是第三方 consumer 最应该从这里开始的接口。

DummyConsumer 是这条路径的维护示例。它读取 hidden rows，把一小段 hidden 异步搬到
CPU，并在结束时 drain worker。它适合作为分析、导出、日志、CPU 侧实验的模板。

## Device-lease 路径

高级写法是：

```python
ConsumerFlow(
    reads=(...),
    writes=(),
    window="out_of_band",
    delivery="device_lease",
    ownership="runtime_lease",
    bundle_key=("engine_step_id", "phase"),
)
```

`out_of_band` 是中性的 step-end 异步窗口，适合不应该阻塞主推理流的 GPU work。

当前 `device_lease` 实现的范围是刻意收窄的：它支持
`bundle_key=("engine_step_id", "phase")` 的 step-scope decode delivery，读取类型
限于 `residual_stream` 和可选的 `request_meta`。它是未来更完整 GPU consumer lane 的
contract 边界，不代表所有 port 类型现在都已经有 durable staged buffers。

Runtime 可以把请求到的 GPU tensor 放进一个 lease 对象：

```python
lease = bundle.entries.get("device_lease")
source = lease.entries.get("source")
target = lease.entries.get("target")
```

当前实现里，`ownership="runtime_lease"` 的意思是这些 tensor entries 由 runtime
持有，consumer 必须按只读数据使用。它们只保证在本次 `consume_bundle()` 调用期间
有效，lease 会用 `lifetime="consume_call"` 显式表达这一点。如果 consumer 要跨
feedback 或 synchronize 边界保留数据，必须先拷贝到自己的 buffer；除非未来 lease
contract 明确提供 durable buffers 和 readiness fences。

只有同时满足这些条件时，才应该考虑这个模式：

- consumer 每个 decode step 或几乎每个 step 都要继续处理 GPU tensor
- 避免额外 staging copy 对吞吐有实际意义
- consumer 有清晰的同步边界，例如 `on_step_end()` 或 `synchronize()`
- 单元测试覆盖了 flow 声明和 lease 形状的 bundle 输入

高级 flow 还可以声明行形状：

```python
ConsumerFlow(
    ...,
    delivery="device_lease",
    row_compaction="first_per_prompt",
)
```

这个设置不会改变全局 decode-step 状态，只会改变投递给这条 flow 的 bundle 形状。
`first_per_prompt` 适合每个 prompt 做一次工作的 consumer，而不是每个 expanded sample
都做一次。当 request metadata 以 `RowBatchMeta` 投递时，runtime 会在 `row_ids`
中记录 compact row 原本对应的 full decode row 位置。metadata 的行数必须匹配投递的
live rows；多余的 tensor 容量只属于 lease，不属于 metadata。

如果 compact row ids 正好是等差 stride，runtime 可以投递一个 runtime-owned 的
strided view，而不是先 materialize 一个 `index_select` 结果。这只是同一个
`device_lease` 生命周期规则下的实现优化；consumer 不应该依赖某个固定的拷贝次数。

## DummyConsumer 和 ESamp 的区别

DummyConsumer 教的是普通接口。它使用 `delivery="bundle"` 和
`ownership="borrowed"`，因为它的目标是展示最安全、最常用的 contract。

ESamp 教的是高级接口。它读取 source/target hidden，调度 runtime adaptation work，
并且可以提供 sampler guidance。训练机制只是这个设计的一部分。因为这些 tensor 留在
GPU 上，并且会交给 step 级 engine 使用，ESamp 选择 `device_lease`。当 ESamp 使用
model-bank 状态时，它还会请求 `first_per_prompt` row compaction，因为 model-bank
更新是每个 prompt 一次，而不是每个 expanded sample 一次。

边界要记清楚：

- runtime 暴露的是通用投递契约
- ESamp 因为 workload 需要，选择了高级契约
- 其他 consumer 默认继续用普通 bundle 路径

## 兼容性

新增 device-lease flow 不会改变已有 consumer。只要 flow 没有显式 opt in，runtime
仍然按普通 bundle entries 投递。复杂 consumer 也可以保留直接 tensor entry 的 fallback；
ESamp 就保留了这条路径，方便单元测试和手动构造 bundle 的集成代码。

拿不准时，先从 DummyConsumer 和普通 bundle 路径开始。只有 profiling 证明 bundle
组装或 staging copy 真的是成本来源，再迁移到 `device_lease`。

对 ESamp 来说，最近的 profiling 确实看到了这个形状：还没启用训练 work 时，
tap-only 配置就已经明显低于 dispatch-off baseline。这类数字属于 ESamp 的优化历史，
不属于通用 contract。通用结论要窄一些：只有 profiling 证明 borrowed bundle delivery
或 consumer-local staging 是瓶颈时，才使用 `device_lease`。

Profiling 也可能证明相反的事情。较长的 Python 片段本身不会同步 CUDA，但会推迟下一步
vLLM enqueue，看起来就像 overlap 被吃掉了。额外的 side-stream kernel 也会和主 decode
kernel 竞争 SM、L2 和显存带宽。遇到这种情况，少一次拷贝有帮助，但不够；consumer 需要
减少热路径 hook 操作、减少每 step Python 调度，或者重新设计 stream 调度。

## 相关文档

- [写你的第一个 Consumer](../getting-started/write-your-first-consumer.md)
- [案例：ESamp 的 Consumer 设计](esamp-design.md)
- [Producer/Consumer 契约](../reference/producer-consumer-contract.md)
- [性能基准测试](benchmarking.md)
