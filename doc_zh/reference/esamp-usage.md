# ESamp 用法与参数

这篇文档是 ESamp 的**命令行参考手册**。它不解释 ESamp 的工作原理，只告诉你有哪些命令、每个参数什么意思、输出怎么看。

**阅读前提**：你已经知道 ESamp 是什么、为什么需要它。如果还不了解，先读 [运行一个内置 Consumer](../getting-started/run-consumer.md)。

这篇文档覆盖三类用法：

1. **功能验证** —— 确认 ESamp 训练路径能跑通、loss 会出现
2. **吞吐 benchmark** —— 测量 consumer 对推理速度的影响
3. **采样干预** —— 开启 distiller 对生成结果的干预

你可以按自己的目的跳读到对应小节。

## 功能验证

在测吞吐之前，必须先确认 ESamp 训练路径真的在工作。如果 loss 根本不出现，benchmark 的结果没有意义。

```bash
python -m tllm.workflows.repro.repro_esamp_loss \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-file test/prompt_debug_list.txt \
  --source-layer-path model.model.layers[0] \
  --target-layer-path model.model.layers[-1]
```

参数说明：

| 参数 | 值 | 含义 |
|------|-----|------|
| `--model-name` | `Qwen/Qwen2.5-0.5B-Instruct` | 选小模型是因为加载快。大模型的行为逻辑相同 |
| `--prompt-file` | `test/prompt_debug_list.txt` | 用预设的 prompt 列表，避免自己构造输入 |
| `--source-layer-path` | `model.model.layers[0]` | 捕获第 0 层的 hidden 作为 distiller 输入。浅层通常包含足够的上下文信息 |
| `--target-layer-path` | `model.model.layers[-1]` | 捕获最后一层的 hidden 作为训练目标。distiller 学习从前者预测后者 |

输出中会打印每步的 loss。确认 `loss_count > 0` 且 loss 值在合理范围内。如果等于 0 或 NaN，说明训练路径有问题，不需要继续测吞吐。

常见失败原因：
- `source-layer-path` 或 `target-layer-path` 写错了（模型结构不同，路径可能不一样）
- GPU 显存不足，ESamp 训练没有启动
- vLLM 版本不兼容，hook 没有正确安装

## 吞吐 benchmark

功能验证通过后，跑标准 benchmark 看吞吐影响。

```bash
VLLM_USE_FLASHINFER_SAMPLER=1 \
python -m tllm.workflows.benchmarks.per_request_esamp_benchmark \
  --emit-json-summary \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 512 \
  --benchmark-batch-size 8 \
  --benchmark-max-new-tokens 256 \
  --benchmark-warmup-rounds 1 \
  --benchmark-rounds 2 \
  --benchmark-ignore-eos \
  --benchmark-disable-prefix-caching \
  --sampling-n 16 \
  --sampling-temperature 0.8 \
  --sampling-top-p 0.95 \
  --sampling-top-k -1 \
  --distiller-lr 1e-3 \
  --model-bank-flush-interval 1 \
  --model-bank-init-method ffn_fast_svd \
  --trajectory-topk 1 \
  --model-bank-train-cudagraph \
  --run-model-bank-case
```

参数分组说明（完整解释见 [运行一个内置 Consumer](../getting-started/run-consumer.md) 的 benchmark 参数表）：

- **模型与显存**：控制模型、精度、显存分配
- **Benchmark 行为**：控制 batch size、生成长度、warmup/rounds
- **采样配置**：temperature、top-p、top-k、并行采样数 `n`
- **ESamp 训练配置**：学习率、model-bank 参数、CUDA graph

真正关心的指标是相对比例：

```
ratio = model_bank_on / single_off
```

`single_off` 是 vanilla vLLM baseline。`model_bank_on` 是启用 ESamp 后的吞吐。

读结果时至少确认：

| 指标 | 合格标准 | 不合格时怎么办 |
|------|---------|--------------|
| `loss_count` | **必须 > 0** | 检查 consumer 是否注册、显存是否够、路径是否正确 |
| `loss_avg` | 在合理范围（如 0.1~10）| 太大可能学习率过高或初始化有问题；太小可能没有真正训练 |
| `single_off` | 和裸 vLLM 的吞吐接近 | 如果明显偏低，说明环境或配置有问题 |
| `model_bank_on` | 低于 `single_off`，但不应差太多 | 如果差一个数量级，检查是否有 CPU sync 或热路径阻塞 |
| `ratio` | 看同一环境下的相对比例 | 低于预期时，先检查 `loss_count`、CPU sync、sampler backend、是否启用 model-bank CUDA graph。7B + min-p 优化路径已经验证过 95%+ 目标区间，小模型或未优化路径可能更低 |

### 为什么只看 ratio 就够了

不同 GPU、不同驱动版本、不同 CUDA 版本的绝对吞吐可能差几倍。但 ratio 衡量的是"同一环境下启用 consumer 前后的相对变化"，排除了环境差异。所以：

- 不要拿你的 `model_bank_on` 和别人比
- 不要拿你的 `single_off` 和论文里的数字比
- 只关心 `model_bank_on / single_off`

## 采样干预

如果你想让 distiller 的预测结果影响生成，在 benchmark 命令后面追加：

```bash
  --enable-distiller-intervention \
  --distiller-beta 0.1 \
  --distiller-sampler-backend post_filter_exact
```

参数说明：

| 参数 | 值 | 含义 |
|------|-----|------|
| `--enable-distiller-intervention` | | 开启 distiller 对采样的干预 |
| `--distiller-beta` | `0.1` | 干预强度。从 0.1 开始，根据效果调整。越大 distiller 影响越强 |
| `--distiller-sampler-backend` | `post_filter_exact` | 只在 LLM 过滤后的候选 token 上应用干预。避免完整词表修饰的计算开销 |

干预公式：

```
new_logit = (1 + beta) * llm_logit - beta * distiller_logit
```

为什么选 `post_filter_exact`：
- vLLM 的 sampler 先执行 temperature / top-k / top-p / min-p 过滤，把词表从几万个 token 缩小到几十~几百个候选
- `post_filter_exact` 只在这几个候选上应用公式，计算量极小
- 如果选完整词表修饰，每次采样都要对几万个 token 做矩阵乘，吞吐会崩

开启干预后，额外确认 distiller candidate stats 非零，说明干预确实生效了。

## 可选 Triton grouped backend

默认 model-bank forward backend 是 `torch`。CUDA / Qwen 吞吐实验可以额外试：

```bash
  --model-bank-forward-backend triton_grouped
```

它只影响 no-grad model-bank prediction / sampling 快路径。训练和 autograd 仍走 torch。

什么时候试这个：
- 你已经跑通了 torch backend 的 benchmark
- 想看看 triton grouped kernel 能不能进一步提升 prediction 路径的吞吐
- 如果报错或数字反而变差，回到默认的 torch backend

## 参数速查

| 参数 | 含义 | 默认值/建议 | 什么时候需要改 |
|------|------|-------------|--------------|
| `--source-layer-path` | distiller 读取输入 hidden 的层 | `model.model.layers[0]` | 如果你想用更深层的信息，或模型结构不同 |
| `--target-layer-path` | distiller 读取目标 hidden 的层 | `model.model.layers[-1]` | 如果你想让 distiller 预测中间层而不是最后一层 |
| `--distiller-hidden-dim` | side model 的隐藏层维度 | `256` | 模型大或任务复杂时增大；想减少计算时减小 |
| `--distiller-lr` | side model 的学习率 | `1e-3` | loss 不收敛时调小；收敛太慢时调大 |
| `--model-bank-rank` | model-bank 低秩分解的 rank | `64` | 模型大时增大；想减少参数量和计算时减小 |
| `--model-bank-flush-interval` | 多久执行一次 optimizer step | `1` | 吞吐敏感时增大（但 loss 可能变差）；loss 敏感时保持 1 |
| `--model-bank-init-method` | side model 的初始化方式 | `ffn_fast_svd` | 不同模型结构可能需要换别的初始化 |
| `--model-bank-train-cudagraph` | 对 ESamp distiller update 捕获 CUDA graph | 默认关闭 | 正式 benchmark 建议开启，减少 kernel launch 开销 |
| `--enable-distiller-intervention` | 开启采样干预 | 默认关闭 | 需要 distiller 影响生成时开启 |
| `--distiller-beta` | 干预强度 | `0.1` | 效果太弱时增大；效果太强或输出质量下降时减小 |
| `--distiller-sampler-backend` | 干预的计算方式 | `post_filter_exact` | 除非你知道自己在做什么，否则保持默认 |

## 下一步

- 理解 ESamp 内部怎么工作：[案例：ESamp 的 Consumer 设计](../developer-guides/esamp-design.md)
- 出了问题怎么排查：[调试指南](../developer-guides/debugging.md)
