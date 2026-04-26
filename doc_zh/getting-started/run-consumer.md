# 运行一个内置 Consumer

这篇文档面向**终端用户**：你已经装好了 tLLM，现在想运行一个内置 consumer 来体验在推理过程中插入自定义逻辑。

读完这篇，你应该能：
1. 理解 consumer 的基本概念
2. 知道运行一个 consumer 的通用流程
3. 用 ESamp 作为例子，跑通完整流程

## Consumer 是什么

在 tLLM 中，**consumer** 是一段在 LLM 推理过程中被调用的代码。它在每个生成 step 接收从推理流中捕获的数据（如 hidden states、请求元信息），然后执行你定义的逻辑。

tLLM 内置了几个 consumer，其中功能最完整的是 **ESamp**。这篇文档以 ESamp 为例，展示运行 consumer 的通用流程。其他内置 consumer 的运行方式结构相同，只是配置参数不同。

## 通用运行流程

无论运行哪个 consumer，步骤都一样：

1. **导入 consumer 和配置类**
2. **创建配置实例**
3. **交给 tLLM runtime 管理**
4. **用带 request mapping 的入口调用生成**
5. **生成结束后同步异步任务并读 stats**

下面以 ESamp 为例走一遍。

## 示例：运行 ESamp

ESamp 是一个在生成过程中做 runtime adaptation 和 sampler guidance 的 consumer。它可以训练一个轻量级网络，用浅层 hidden 预测深层 hidden，并可选地用预测结果干预采样分布。

### 最小示例

```bash
python starter.py
```

这会：
1. 加载 `Qwen/Qwen2.5-7B-Instruct`
2. 创建 `ESampConsumer` 实例
3. 把 consumer 注册到 tLLM runtime
4. 并行生成 16 条回答
5. 在生成过程中运行 ESamp 的训练机制

输出末尾会有 consumer 的统计信息：`loss_count` 和 `loss_avg`。如果 `loss_count > 0`，说明 ESamp 的训练机制确实运行了。

缩短输出：

```bash
python starter.py --max-new-tokens 32
```

### 关键代码结构

打开 `starter.py`，核心结构如下。通用概念是显式创建 `ESampConsumer(...)`，再通过 `register_consumer(...)` 注册；`starter.py` 里的 workflow helper 只是为了让 demo 和 benchmark 少写样板代码。

```python
from vllm import SamplingParams

from tllm import make_llm, register_consumer
from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.runtime import residual_runtime as runtime
from tllm.workflows import esamp_support

# 1. 配置 ESamp，并把 consumer 交给 runtime
consumer = ESampConsumer(ESampConsumerConfig(
    graph_scratch_rows=64,
    source_layer_path="model.model.layers[0].input_layernorm",
    target_layer_path="model.model.layers[-1].input_layernorm",
    enable_esamp_training=True,
    distiller_hidden_dim=128,
    distiller_lr=1e-3,
    per_request_model_bank=True,
    model_bank_slots=16,
    model_bank_rank=64,
    model_bank_flush_interval=1,
    model_bank_train_cudagraph=True,
    enable_distiller_intervention=True,
    distiller_beta=0.1,
    distiller_sampler_backend="post_filter_exact",
))
register_consumer(consumer)

# 2. 创建 vLLM 实例。tLLM 会在这里安装 vLLM v1 runtime patch
llm = make_llm(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.8,
    max_model_len=512,
    enable_prefix_caching=False,
    enforce_eager=False,
    seed=2026,
)

# 3. 显式构造 16 个并行请求。starter.py 这样写，是为了绕开部分 vLLM V1 版本中 n>1 输出不稳定的问题
prompts = ["用两句话介绍 tLLM。"] * 16
params = [
    SamplingParams(n=1, temperature=0.8, top_p=0.95, max_tokens=32)
    for i in range(16)
]

outputs = esamp_support.run_generate_with_request_mapping(
    llm,
    prompts,
    params,
    request_prompt_indices=[0] * 16,
    request_sample_indices=list(range(16)),
)

# 4. 排空 ESamp 异步队列，读取统计
runtime.synchronize_esamp()
stats = runtime.read_and_reset_esamp_stats(sync=True)
print(stats)
```

`starter.py` 默认对所有显式 request 使用同一个 seed：

```bash
python starter.py --seed 2026 --seed-mode shared
```

这样可以避开 vLLM 的 per-request generator 路径，在环境支持时让 FlashInfer sampler 继续生效。如果你更需要每条 answer 都有独立且可复现的随机流，可以使用：

```bash
python starter.py --seed 2026 --seed-mode per-request
```

shared 模式会把 `seed` 交给 LLM engine，request 级别的 `SamplingParams.seed` 保持未设置。per-request 模式会使用 `seed + i` 作为每个 request 的 seed。此时 vLLM 可能打印 `FlashInfer 0.2.3+ does not support per-request generators. Falling back to PyTorch-native implementation.` 这是该 seed 模式下的预期 warning。

对其他 consumer 来说，结构也是类似的：配置 consumer → 交给 runtime → 生成 → 同步 → 读 stats。区别在于 ESamp 额外需要 request mapping 和 sampler intervention 配置，所以 starter 会比最简单的只读 consumer 多几行。

### 跑 benchmark

如果你想定量地知道 ESamp 对推理吞吐有多大影响，跑标准 benchmark：

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

这个命令很长，但它由几个逻辑组构成。理解每组参数的作用，你才能根据自己的硬件和场景调整：

**环境变量**
- `VLLM_USE_FLASHINFER_SAMPLER=1`：开启 FlashInfer 采样后端。它对推理吞吐有明显提升，建议在吞吐实验中统一开启。如果编译失败，设为 `0`。

**模型与显存**
| 参数 | 值 | 为什么这样设 |
|------|-----|-------------|
| `--model-name` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B 是小模型，加载快、适合快速验证。换成更大的模型时数字会变，但 ratio 的观察方法不变 |
| `--dtype` | `bfloat16` | 当前主流 GPU 支持，比 fp16 更稳定 |
| `--gpu-memory-utilization` | `0.5` | vLLM 最多占 50% 显存，剩下给 ESamp 训练机制。如果显存充裕可以提高到 0.7~0.8 |
| `--max-model-len` | `512` | 限制序列长度。如果 `sampling_n` 高或 batch 大，可能需要增大到 1024 |

**Benchmark 行为**
| 参数 | 值 | 为什么这样设 |
|------|-----|-------------|
| `--benchmark-batch-size` | `8` | 8 个请求并行。batch size 越大吞吐越高，但 ESamp 训练压力也越大 |
| `--benchmark-max-new-tokens` | `256` | 每个请求最多生成 256 个 token。decode step 越多，ESamp 训练机制的总训练量越大 |
| `--benchmark-warmup-rounds` | `1` | 先跑 1 轮 warmup，排除冷启动和 CUDA cache 的影响 |
| `--benchmark-rounds` | `2` | 正式跑 2 轮取平均。数字小是因为主要用来验证功能，正式报告建议 5~10 轮 |
| `--benchmark-ignore-eos` | | 忽略 EOS，强制生成满 256 个 token。这样不同 run 之间生成长度一致，结果可比 |
| `--benchmark-disable-prefix-caching` | | 关闭前缀缓存，避免缓存命中干扰吞吐测量 |
| `--emit-json-summary` | | 输出机器可读的 JSON 摘要，方便后续分析 |

**采样配置**
| 参数 | 值 | 为什么这样设 |
|------|-----|-------------|
| `--sampling-n` | `16` | 每个 prompt 并行采样 16 条。`n` 越大，ESamp 训练机制处理的行越多，吞吐压力越大 |
| `--sampling-temperature` | `0.8` | 标准 temperature |
| `--sampling-top-p` | `0.95` | nucleus sampling 阈值 |
| `--sampling-top-k` | `-1` | 不限制 top-k |

**ESamp 训练机制配置**
| 参数 | 值 | 为什么这样设 |
|------|-----|-------------|
| `--distiller-lr` | `1e-3` | side model 的学习率。这个值对收敛速度影响大，但不是吞吐的主要瓶颈 |
| `--model-bank-flush-interval` | `1` | 每 1 个 step flush 一次。interval 越大吞吐越高（减少 optimizer step 频率），但 loss 可能变差 |
| `--model-bank-init-method` | `ffn_fast_svd` | Qwen 模型推荐的初始化方式。不同模型结构可能需要换别的 |
| `--trajectory-topk` | `1` | 每个 step 只保留 top-1 轨迹用于训练 |
| `--model-bank-train-cudagraph` | | 对 ESamp distiller update 的 forward/backward 捕获 CUDA graph，减少 kernel launch 开销 |
| `--run-model-bank-case` | | 只跑 model-bank 模式。如果不加，会同时跑 single、per-request、model-bank 三种模式 |

### 读结果

跑完后看 JSON 输出或终端摘要。真正重要的指标是相对比例：

```
ratio = model_bank_on / single_off
```

| 指标 | 含义 | 怎么判断 |
|------|------|---------|
| `single_off` | vanilla vLLM baseline 吞吐 | 先看这个确认 baseline 正常 |
| `model_bank_on` | 启用 ESamp 后的吞吐 | 看绝对值是否合理 |
| `ratio` | 相对开销 | 核心指标。这个值取决于模型大小、min-p/top-p 配置、是否开启 distiller intervention、model-bank graph replay 等因素；在 7B + min-p 优化路径上，ESamp model-bank 已经能达到 95%+ 的目标区间 |
| `loss_count` | **必须大于 0** | 等于 0 说明训练没发生，不管吞吐多高都是失败 |
| `loss_avg` | 训练目标的平均值 | 应该在合理范围（如 0.1~10），太大或太小都可能有问题 |

如果 `loss_count == 0`，先检查 consumer 是否注册成功、`source-layer-path` 和 `target-layer-path` 是否有效、GPU 显存是否足够。

## 从 ESamp 推广到其他 Consumer

ESamp 是内置 consumer 中最复杂的一个。如果你跑通了 ESamp，其他 consumer 的运行方式结构完全相同：

1. 导入对应的 consumer 类和配置类
2. 创建配置实例（参数不同）
3. 注册到 runtime
4. 生成后读 stats

不同 consumer 的详细参数见各自的参考文档。

## 下一步

- 想深入了解 ESamp 的内部设计：[案例：ESamp 的 Consumer 设计](../developer-guides/esamp-design.md)
- 想了解 ESamp 的完整参数列表：[ESamp 用法与参数](../reference/esamp-usage.md)
- 想理解普通接口和高级接口的区别：[Consumer 投递模式](../developer-guides/consumer-delivery-modes.md)
- 想写自己的 consumer：[写你的第一个 Consumer](write-your-first-consumer.md)
