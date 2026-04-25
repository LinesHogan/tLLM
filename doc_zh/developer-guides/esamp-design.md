# 案例：ESamp 的 Consumer 设计

这篇文档是**案例研究**，不是必读材料。

如果你只想运行 ESamp，去读 [运行一个内置 Consumer](../getting-started/run-consumer.md) 就够了。这篇面向的是**想写一个复杂 consumer 的开发者**——通过一个真实案例，展示 tLLM consumer 框架的能力边界，以及复杂需求应该如何拆分到框架提供的机制上。

## 这个 Consumer 要解决什么问题

ESamp 是一个在 LLM 生成过程中做 **test-time training** 的 consumer。它想在每个 decode step 做这些事：

1. 拿到某个浅层的 hidden state（输入）
2. 拿到某个深层的 hidden state（训练目标）
3. 在线训练一个轻量级 distiller，学习从前者预测后者
4. 可选地，用 distiller 的预测结果干预下一个 token 的采样分布

这些需求同时涉及：数据捕获、异步重计算、有状态训练、采样干预。它是一个很好的例子，用来展示 consumer 框架能支持到什么复杂度。

## Consumer 框架提供的机制

ESamp 没有自己实现数据捕获或采样管道插入。它完全依赖 consumer 框架提供的通用机制：

### 数据捕获：声明式 Port 读取

ESamp 需要 source hidden、target hidden 和请求元信息。它在 `flows()` 里声明：

```python
ConsumerFlow(
    reads=(
        ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
        ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
        RequestMeta.read(),
    ),
    ...
)
```

Runtime 负责：安装 layer hook、从 packed tensor 中定位行、按请求组装 bundle。ESamp 只关心 `bundle.entries["source"]` 和 `bundle.entries["target"]` 里有没有数据。

### 异步训练：out_of_band_train window

Side-train 涉及矩阵乘和 optimizer step，不能在主推理流里同步做。ESamp 把 `window` 设为 `"out_of_band_train"`，表示这条 flow 是训练专用的异步路径：runtime 负责捕获 source/target hidden，ESamp 负责把它们送进 side stream 上的训练引擎。

```python
ConsumerFlow(
    ...,
    window="out_of_band_train",
)
```

### 采样干预：Sampler Provider

ESamp 想修改候选 token 的 logits。但 ESamp 自己不直接 patch vLLM 的 sampler。它实现一个 `SamplerModifierProvider`，交给 tLLM 的通用 sampler bridge：

- Runtime 在 sampler 阶段构造候选 token 的 view
- Bridge 调用所有已注册的 provider
- Provider 返回候选级的 logits delta
- Runtime 应用 delta 后再执行最终采样

这里的分工很重要：patch vLLM `compute_logits` 和 sampler 是 tLLM runtime 的职责；ESamp 只提供“给定候选 token，如何修饰 logits”的算法逻辑。这样以后别的 consumer 也可以复用 sampler bridge，而不需要复制 ESamp 的内部代码。

### 有状态训练：Consumer 内部状态管理

Side-train 需要维护模型参数、optimizer 状态、每个请求的 slot 映射。这些状态完全在 consumer 内部管理，runtime 不感知。

ESamp 展示了三种典型的状态管理策略：

| 模式 | 状态策略 | 适用场景 |
|------|---------|---------|
| **Single** | 全局共享一组参数 | 快速验证，请求少 |
| **Per-request** | 每个请求独立参数 | 语义直观，请求数固定且少 |
| **Model-bank** | 固定数量的 slot，动态分配给活跃请求 | 高并发，推荐路径 |

这三种模式不是 ESamp 特有的，而是任何需要在请求维度维护状态的 consumer 都会面临的典型选择。

## ESamp 的组件拆分

一个复杂的 consumer 应该把框架交互层和算法实现层拆开。ESamp 的拆分方式如下：

| 组件 | 职责 | 与框架的关系 |
|------|------|-------------|
| `ESampConsumer` | 声明 ports、接收 bundle、触发训练 | 面向 `ConsumerFlow` 和 `PortBundle` |
| `ESampTrainEngine` | 维护参数、forward/backward、model-bank 调度 | 纯内部实现，不依赖框架 |
| `ESampSamplerModifierProvider` | 把 distiller 输出转成 logits delta | 面向通用 sampler bridge |

Runtime 只认识 `ESampConsumer` 的公开 consumer 接口（`flows()`、`consume_bundle()`、`synchronize()`，以及 ESamp 为训练收尾实现的 `apply_feedback()`），还有 `ESampSamplerModifierProvider` 的 sampler 接口。`ESampTrainEngine` 对 runtime 完全不可见。

## Decode Step 时序：Consumer 视角

一次 decode step 中，从 ESamp 的视角看，事件顺序是：

1. **Layer hook 捕获 source hidden** —— runtime 只做定位和 staging，不在 hook 里启动复杂计算
2. **Layer hook 捕获 target hidden** —— target hidden 作为训练目标进入同一个 step 的 bundle
3. **LLM 进入 `compute_logits`** —— tLLM runtime 在这个边界触发 distiller no-grad precompute。这个时机比 sampler 早，又比 layer hook 更适合和 vLLM graph/compile 路径共存
4. **Sampler bridge 调用 Provider** —— ESamp 根据 LLM 已过滤出的候选 token，计算 distiller logits，并按公式修饰候选 logits
5. **vLLM 在修饰后 logits 上采样**
6. **`execute_model.post` / side-train window** —— runtime 分发 bundle，ESamp 执行 delayed backward / model-bank flush

这里的关键观察是：consumer 框架允许你把逻辑拆到多个 hook 点（layer output、sampler、step end），而不是把所有事情塞在一个回调里。

尤其要避免把 distiller forward 直接放进 PyTorch forward hook。hook-time 计算看起来最早，但在 vLLM compile / CUDA graph 路径下容易产生不稳定行为；当前实现把真正需要喂给 sampler 的 precompute 推迟到 `compute_logits` 边界，这是更稳妥的折中。

## 性能原则：适用于所有 Consumer

ESamp 的调优经验中，最有复用价值的部分不是 ESamp 特有的参数组合，而是对所有 consumer 都成立的通用原则：

- `loss_count == 0` 永远不是成功。吞吐再高，如果 consumer 没真正拿到数据或触发计算，都是失败
- 不要在热路径里做 `.item()`、`.tolist()`、大规模 `.cpu()`。这些操作隐式同步 GPU
- 不要在 `consume_bundle()` 里启动复杂 Python 调度或同步 drain worker
- 可向量化的工作不要放在 Python 循环里
- 状态管理策略（single/per-request/model-bank）的切换，往往比微优化代码更有影响

## 从 ESamp 到你自己的 Consumer

ESamp 展示了 consumer 框架的上限：你可以同时做数据捕获、异步训练、采样干预、有状态管理。但大多数 consumer 不需要这么复杂。

如果你要写一个新的 consumer，建议从简单需求开始：

1. 先用 `ConsumerFlow` 声明你要什么数据
2. 在 `consume_bundle()` 里只读不做复杂计算
3. 确认 `read_stats()` 能看到非零计数
4. 然后再逐步加入：异步 worker、状态管理、采样干预

ESamp 是一个参考实现，不是模板。你不需要复制它的组件拆分，只需要理解框架提供了哪些机制，以及每个机制解决什么问题。
