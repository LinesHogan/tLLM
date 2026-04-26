# 调试 Consumer

调试 tLLM consumer 时，核心原则是**先确认数据有没有到、语义有没有对，再 profile**。不要一上来就猜性能瓶颈。

这篇文档提供一个分层的调试方法论：

1. **先拆边界** —— 问题出在 Producer（数据提取）、Runtime（hook 和 bundle）还是 Consumer（你的逻辑）？
2. **再按症状查** —— consumer 没被调用、bundle 字段不对、吞吐暴跌、OOM 等常见症状各有排查路径

## 三个边界

| 边界 | 负责什么 | 典型问题 |
|------|---------|---------|
| Producer | 从 vLLM packed tensor 定位 rows | hidden 错位、MSE 失败 |
| Runtime | hook、port frame、bundle 组装、dispatch | consumer 没被调用、bundle 缺字段 |
| Consumer | 你的算法逻辑 | 队列不 drain、stats 异常、CPU/GPU sync |

定位到边界后，再深入。下面按症状展开。

## Consumer 没被调用

按这个顺序查：

1. `flows()` 是否返回了至少一个 `ConsumerFlow`
2. `reads` 里的 `phase` 是否匹配当前生成阶段（prefill 还是 decode）
3. `role` 是否和 `bundle.entries[...]` 里的 key 一致
4. `bundle_key` 是否把你需要的数据聚合到同一个 bundle
5. consumer 是否真的注册到了 runtime（`register_dispatch_consumer` 是否被调用）
6. consumer 是否被 `set_enabled(False)` 关闭了

可以临时在 `consume_bundle()` 开头加一个计数器，然后在生成后读 `read_stats()` 验证。

## Bundle 有字段但值不对

先确认你读的是 view 还是 copy。`residual_stream` 给到的 hidden 通常是 runtime buffer 的 view：

- 当前调用内只读：可以直接用
- 跨 step 使用：必须 clone
- 交给 CPU worker：用 non-blocking staging
- 不要原地写，除非你实现的是明确的 write-back port

如果怀疑 producer localization 错了，去 [正确性验证](validation.md) 跑 decode/prefill MSE。

## CPU/GPU 同步问题

当你的 consumer 吞吐突然变差时，先搜这几个模式：

```bash
rg -n "\\.item\\(|\\.tolist\\(|\\.cpu\\(|synchronize\\(|print\\(" tllm
```

常见坑：

- 每 step 热路径里 `.item()` 读 GPU 标量
- 对 GPU tensor `.tolist()`
- 每步 `.cpu()` 拷完整 tensor
- `consume_bundle()` 内同步 drain CPU worker
- 每 step print 大量日志

Debug 时可以临时 print；提交前把热路径打印删掉，或放到低频 summary。

## Async CPU worker 不 drain

如果 consumer 像 DummyConsumer 一样使用异步 CPU worker：

- `consume_bundle()` 只 enqueue
- `on_step_end()` 可以按 interval drain
- `synchronize()` 必须 drain 剩余队列
- 队列满时不要在热路径强制 drain。优先 drop 或把 backpressure 推到非热路径

## Sampler / guidance 不生效

先分清两层：

1. provider 是否 active
2. sampler patch 是否调用了 provider

如果是 distiller / ESamp 路径，先看 [ESamp 用法与参数](../reference/esamp-usage.md)。

通用的 sampler consumer 检查清单：

- 是否声明或注册了 sampler provider
- provider 的 `is_active()` 是否返回 true
- request rows 是否和 logits rows 对齐
- 修改后的 logits 是否只作用于预期候选

## OOM

常见处理：

- 模型加载 OOM：降低 `--gpu-memory-utilization`
- 高 `sampling_n` OOM：增大 `--max-model-len` 或降低 `n`
- consumer 暂存过多：降低 queue size，按 step stride export，或 drop
- ESamp training OOM：降低 `--distiller-hidden-dim` 或 `--model-bank-rank`

vLLM 异常退出后，先检查 stale worker：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

## 下一步

- 验证数据对不对：[正确性验证](validation.md)
- 测吞吐影响：[性能基准测试](benchmarking.md)
- 理解代码先看哪：[代码结构](../reference/project-structure.md)
