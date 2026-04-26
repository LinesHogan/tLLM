# 写你的第一个 Consumer

这篇教程面向**Consumer 开发者**：你想在推理过程中插入自己的代码。

读完这篇，你应该能：
1. 理解 `ConsumerFlow` 怎么声明数据需求
2. 知道 `consume_bundle()` 里能拿到什么、不能做什么
3. 学会把数据异步搬到 CPU，不阻塞推理
4. 在生成结束后正确 flush

## 前提

你需要先读过 [架构详解](../developer-guides/architecture.md)，知道 PortBundle、Producer、Runtime 这几个角色是干什么的。如果你还没读，现在去读。

## DummyConsumer 是做什么的

`tllm/consumers/dummy/` 是一个最小可运行的 consumer。它：
- 声明读取 `residual_stream` 和 `request_meta`
- 在 `consume_bundle()` 里把 hidden state 非阻塞地搬到 CPU
- 在 CPU worker 里做一点点 demo 处理（注入噪声、打印摘要）
- 队列满时直接丢弃，不在热路径里阻塞

这个 consumer 不解决实际问题，但它展示了一个健康的异步模式。

## 第一步：复制模板

```bash
cp -r tllm/consumers/dummy tllm/consumers/my_consumer
```

把里面的 class 名从 `DummyConsumer` / `DummyConsumerConfig` 改成你自己的名字，改 `consumer_id`。

## 第二步：声明你要什么数据

Consumer 的核心入口是 `flows()` 方法。它返回一组 `ConsumerFlow`，告诉 runtime：
- 我要读哪些 port
- 我要写哪些 port（如果有）
- 在哪个 window 执行
- bundle 怎么聚类

例子：读取 layer 0 的 decode hidden state，加上 request 元信息。

```python
from tllm.ports.base import ConsumerFlow
from tllm.ports.residual_stream import ResidualStream
from tllm.ports.request_meta import RequestMeta
from tllm.ports.cpu_export import CpuExport

def flows(self):
    return [
        ConsumerFlow(
            reads=(
                ResidualStream.read(
                    layer=0,
                    site="block_output",
                    phase="decode",
                    role="hidden",
                ),
                RequestMeta.read(),
            ),
            writes=(
                CpuExport.write(channel="my_consumer", format="row_batch"),
            ),
            window="background",
            bundle_key=("engine_step_id", "phase"),
        )
    ]
```

几个关键字段：
- `role`：在 `bundle.entries` 里的 key 名。这里 `"hidden"` 意味着 `bundle.entries["hidden"]` 能拿到数据
- `window="background"`：异步执行，不阻塞主推理流。如果你的 consumer 要改 logits 影响采样，用 `"same_step"`
- `bundle_key`：runtime 用哪些字段把多个 port 的数据聚合成一个 bundle。通常 `"engine_step_id"` + `"phase"` 就够了
- 默认投递模式是 `delivery="bundle"`，entries 按 borrowed view 使用。除非 profiling
  证明你的 consumer 需要 device-lease 投递，否则保持这个默认值。

## 第三步：接收并处理数据

`consume_bundle(bundle, ctx)` 是数据到达时的回调。

```python
def consume_bundle(self, bundle, ctx):
    hidden = bundle.entries.get("hidden")
    if hidden is None:
        return
    self.num_rows += int(hidden.shape[0])
```

注意：
- `bundle.entries["hidden"]` 通常是 runtime buffer 的 **view**，不是拷贝
- 如果你要跨 step 保存它，必须 `hidden.clone()`
- 不要在这里做重 CPU 计算，不要 `.item()`、`.tolist()`、大规模 `.cpu()`

## 第四步：异步搬到 CPU（推荐模式）

如果你的处理逻辑主要在 CPU 上，推荐用这个结构：

```python
def consume_bundle(self, bundle, ctx):
    hidden = bundle.entries.get("hidden")
    if hidden is None:
        return
    if self.worker.pending() >= self.config.max_queue_size:
        self.dropped += 1
        return

    def submit():
        hidden_cpu = hidden.detach().to("cpu", non_blocking=True)
        self.worker.enqueue(hidden_cpu)

    self.stream_runtime.run(submit)

def synchronize(self):
    self.stream_runtime.synchronize()
    self.worker.drain()
```

`stream_runtime` 是一个辅助对象，帮你管理自定义 CUDA stream 上的异步操作。

关键点：
- `consume_bundle()` 只做 enqueue，不做处理
- 真正的 CPU 工作在 `worker.drain()` 里执行
- `synchronize()` 在生成结束时调用，把队列排空

## 第五步：注册到 Runtime

在你的入口代码里：

```python
from tllm import register_consumer
from tllm.consumers.my_consumer import MyConsumer, MyConsumerConfig

consumer = MyConsumer(MyConsumerConfig())
register_consumer(consumer)

# 然后照常调用 vLLM generate
outputs = llm.generate(prompts, sampling_params)

consumer.synchronize()
stats = consumer.read_stats()
```

consumer 必须在 `llm.generate()` 之前注册。否则 runtime 不会为它装 hook。

## 验证你的 Consumer

先跑 DummyConsumer 的回归测试，确保模板本身没被你改坏：

```bash
python -m pytest -q test/test_dummy_consumer_unit.py
```

然后写你自己的单元测试，覆盖：
1. `flows()` 返回了正确的 `ConsumerFlow`
2. `consume_bundle()` 在预期数据到达时被调用
3. `synchronize()` 能排空资源，不会泄露或挂起

## 常见错误

| 现象 | 原因 |
|------|------|
| `consume_bundle` 从没被调用 | `flows()` 没返回 `ConsumerFlow`；或 phase 不匹配；或 consumer 没注册到 runtime |
| bundle 有字段但值不对 | 读的可能是 view 而不是 copy；跨 step 用了 view 导致数据被覆盖 |
| 吞吐突然暴跌 | 热路径里有 `.item()`、`.tolist()`、`.cpu()`、或同步 drain worker |
| 队列不断增长最后 OOM | 没有定期 drain，或 `synchronize()` 没被调用 |

## 下一步

- 想理解普通接口和高级接口的区别：[Consumer 投递模式](../developer-guides/consumer-delivery-modes.md)
- 想测性能影响：[性能基准测试](../developer-guides/benchmarking.md)
- 怀疑数据不对：[正确性验证](../developer-guides/validation.md)
- 出了 bug：[调试指南](../developer-guides/debugging.md)
