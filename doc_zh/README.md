# tLLM

tLLM 是一个在 vLLM v1 推理引擎中插入运行时数据捕获与消费层的框架。它让外部代码在推理过程中读取、分析甚至修改模型内部状态，而不用 fork vLLM。

---

## 为什么需要这个

大模型推理引擎（如 vLLM）为了吞吐做了大量优化：continuous batching、PagedAttention、CUDA Graph、cuda kernel fusion。代价是代码高度封装，想在推理过程中做点额外的事情——比如读取某层 hidden state 做个实时分析，或者边生成边训练一个小网络——你会发现 hook 点藏得很深，直接改代码容易搞崩吞吐，而且每次升级 vLLM 都要重写。

另一边，HuggingFace Transformers 易于开发，但推理效率低，开发出来的算法很难在生产环境落地。

tLLM 试图解决这个问题：保留 vLLM 的推理性能，同时提供一套声明式接口，让你能在不修改 vLLM 核心代码的前提下，在推理过程中捕获数据、执行自定义逻辑。

---

## tLLM 做了什么

tLLM 在 vLLM v1 引擎的三个生命周期方法上插入 patch：

- `load_model`：在模型加载完成后安装 layer forward hook
- `_prepare_inputs`：在每步开始时快照请求定位信息（哪些行是 decode、哪些是 prefill）
- `execute_model`：在推理完成后触发数据分发和 consumer 回调

此外，对于需要干预采样的场景，tLLM 还会 patch `compute_logits` 和 sampler，以支持 distiller guidance 这类算法。

数据从 vLLM 流出后，经过 **Producer** 做定位提取，经过 **Runtime** 做聚合调度，最后以 `PortBundle` 的形式交给你的 **Consumer**。Consumer 通过 `ConsumerFlow` 声明需求，框架负责路由。Consumer 和 vLLM 核心代码之间没有直接耦合。

---

## 你是哪种用户

tLLM 的文档按三种用户类型组织：

### 1. 终端用户：我想用内置算法做生成

你想直接跑 tLLM 内置的 ESamp consumer，看看 runtime adaptation 和 sampler guidance 的效果，或者用它来做生成。

**入口**：
1. [安装指南](getting-started/installation.md)
2. [运行一个 Consumer](getting-started/run-consumer.md)
3. [ESamp 用法与参数](reference/esamp-usage.md)

### 2. Consumer 开发者：我想写自己的逻辑

你想在推理过程中插入自己的代码，比如读取 hidden states 做分析、导出数据、实现 runtime adaptation，或者实现新的 sampler-guidance 算法。

**入口**：
1. [安装指南](getting-started/installation.md)
2. [架构详解](developer-guides/architecture.md) — 理解数据是怎么从 vLLM 里流出来的
3. [写你的第一个 Consumer](getting-started/write-your-first-consumer.md)
4. [Consumer 投递模式](developer-guides/consumer-delivery-modes.md)
5. [Consumer 调试指南](developer-guides/debugging.md)

### 3. Contributor：我想改 tLLM 本身

你想改 runtime、加新 port、或者给 tLLM 提 PR。

**入口**：
1. [代码结构](reference/project-structure.md)
2. [开发者测试指南](development/testing-guide.md)
3. [术语表](reference/glossary.md)

---

## 一句话快速开始

```bash
git clone <repo-url>
cd tLLM
python -m venv .venv
source .venv/bin/activate
pip install vllm
pip install -e .

# 跑 ESamp 示例
python starter.py --max-new-tokens 32
```

如果环境正常，你会看到 16 条生成结果，以及 ESamp 的 `loss_count`、`loss_avg` 和 sampler-guidance counters。starter 默认使用 `--seed-mode shared`；如果你更需要每个 request 独立的 `seed + i` 随机流，可以使用 `--seed-mode per-request`，但这可能触发 FlashInfer sampler fallback warning。

---

## 环境要求

- Python >= 3.10
- vLLM >= 0.7.2，**仅限 v1 引擎**（v0 不兼容）
- PyTorch >= 2.0
- CUDA >= 12.1

当前主要验证的版本是 `vllm==0.10.x`。

---

## 相关链接

- [vLLM 官方文档](https://docs.vllm.ai/)
