# 性能基准测试

这篇文档讲怎么给 consumer 测吞吐，得到"可信"的数字。

benchmark 的核心问题是：**环境差异会让绝对吞吐差几倍**。只看 consumer 开启时的 tok/s 没有意义。正确的做法是跑对照实验，看相对比例。

这篇文档的结构：

1. **核心原则** —— 什么是对照实验，什么指标真正重要
2. **通用流程** —— 任何 consumer benchmark 都遵循的代码结构
3. **Baseline** —— DummyConsumer 的默认配置为什么故意调得很稀疏
4. **性能排查** —— 如果吞吐掉了很多，按什么顺序查

## 核心原则：ratio 比绝对值重要

任何 consumer benchmark 都至少需要两个 case：

- `off`：不启用 consumer，跑 vanilla vLLM
- `on`：启用 consumer，其他参数完全一致

真正关心的指标是：

```
ratio = on_tok_per_s / off_tok_per_s
```

只看 `on` 的绝对 tok/s 没意义。同样的 consumer，在不同 GPU、不同 batch size、不同模型下，绝对数字可能差几倍。

如果你的 consumer 还会训练或修改采样，还要同时看功能统计：loss、processed rows、candidate count 等。`loss_count == 0` 时，不管吞吐多高都不是成功。

## 通用流程

```python
from tllm.runtime import residual_runtime as runtime
from tllm.consumers.my_consumer import MyConsumer, MyConsumerConfig

consumer = MyConsumer(MyConsumerConfig())
runtime.register_dispatch_consumer(consumer)

# 用和 off case 完全相同的 prompts、sampling_params
outputs = llm.generate(prompts, sampling_params)

consumer.synchronize()
stats = consumer.read_stats()
```

注意：
- consumer 注册在生成前
- off/on 的 prompts、sampling params、模型、batch size 必须完全一致
- benchmark 结束后调用 `synchronize()`，把异步 worker drain 干净，再读 stats

## DummyConsumer baseline：框架 overhead 有多小

`DummyConsumer` 的默认配置是故意调得很稀疏的：

- `dispatch_every_n_steps=256`：多数 decode step 不组装 bundle
- `max_bundle_rows=1`：只处理一行样本
- `export_max_cols=16`：只搬运 16 维 hidden slice

这个配置下，在 RTX 4090 + `Qwen2.5-0.5B` + batch=8 + n=16 + max_new_tokens=256 的环境里：

| 模式 | tok/s |
|------|-------|
| vanilla vLLM | 27691 |
| DummyConsumer default | 26934 |

比例约 0.97。读这个结果时还要检查 `processed_batches > 0`、`processed_rows > 0`，确认 consumer 真的拿到了 hidden。

DummyConsumer 的默认配置不是"优化版"，而是"看看框架 overhead 有多小"。你的 consumer 如果配置更密、处理更重，比例自然会下降。

## 第一轮性能排查

如果你的 consumer 让吞吐掉了很多，按这个顺序查：

1. 搜索热路径里的同步点：
   ```bash
   rg -n "\\.item\\(|\\.tolist\\(|\\.cpu\\(|synchronize\\(|print\\(" tllm
   ```
2. 检查 `consume_bundle()` 是否做了重 CPU 工作
3. 是否每步都跨设备拷贝大 tensor
4. 是否频繁 drain worker
5. benchmark workload 和 baseline 是否完全一致

更多排查细节见 [调试指南](debugging.md)。

## 下一步

- 跑 ESamp 的 benchmark：[ESamp 用法与参数](../reference/esamp-usage.md)
- 理解普通和高级投递路径：[Consumer 投递模式](consumer-delivery-modes.md)
- 验证功能没坏：[正确性验证](validation.md)
