# ESamp 优化历程

这份文档记录 ESamp 从“语义正确但很慢的 distiller intervention 原型”
一步步优化到 7B 目标吞吐的过程。

注意：下面的数字不是同一个 workload 的连续排行榜。每一行只应该理解为该
局部实验里的 before/after 对比。

## 背景概念

### ESamp 是什么

ESamp 是 tLLM 内置的 side-training consumer。它在 LLM decode 过程中读取
LLM 某一层的 source hidden，并用一个小模型学习预测另一层的 target hidden。
这个小模型在本文里叫 distiller。

训练目标一直是 hidden prediction loss。换句话说，训练本身没有因为 sampler
intervention 改变。

### Distiller intervention 是什么

开启 intervention 后，distiller 会用当前 source hidden 预测一个 hidden，
再经过和 LLM 相同的 LM head 得到 distiller logits。采样时，LLM 先按自己的
规则过滤候选 token，然后 ESamp 对候选 logits 应用：

```text
new_logit = (1 + beta) * llm_logit - beta * distiller_logit
```

最后在修饰后的 logits 上采样。

### Model-bank 是什么

`model_bank_on` 是 ESamp 的高并发路径。它不是给每个 request 动态创建一个
完整小模型，而是把多个请求映射到固定的 bank slots。每个 slot 有自己的
distiller 参数。这样可以批量训练、复用固定 buffer，并配合 CUDA graph replay。

### 读吞吐数字时看什么

- `single_off`：同一次 benchmark 里的 vanilla vLLM baseline。
- `single_on`：一个共享 side model 的训练路径。
- `per_request_on`：每个 request 一个 side model 的训练路径。
- `model_bank_on`：model-bank 训练路径。
- ratio：默认指 `model_bank_on / single_off`。
- `loss_count > 0`：训练真的发生。吞吐很高但 `loss_count == 0` 是失败。

最终目标 workload 是：

```text
Qwen/Qwen2.5-7B-Instruct
batch=8
sampling_n=16
max_new_tokens=256
top_p=1.0
top_k=-1
min_p=0.05
dtype=bfloat16
```

## 术语补充

### all-random / all-greedy fast paths

sampler bridge 每一步会处理一批 decode rows。每一行可能是随机采样，也可能是
贪心采样：

- random：通常 `temperature > 0`，需要按概率分布采样。
- greedy：通常 `temperature == 0`，直接选最大 logit。

最通用的代码要支持“同一个 batch 里一部分 random、一部分 greedy”的混合情况。
这种通用路径通常要创建 mask、调用 `nonzero()` 找出两组行、分别处理，再把结果
合并回原顺序。

但 benchmark 中常见的是整批 rows 都是 random，或者整批 rows 都是 greedy。
`all-random/all-greedy fast paths` 就是先识别这两种同质 batch：

- 如果全是 random，整批直接走 random 分支。
- 如果全是 greedy，整批直接走 greedy 分支。
- 只有混合 batch 才走通用 mask/nonzero 路径。

这个优化不改变采样结果，只减少常见情况下的 Python/tensor 分组开销和潜在同步点。

### pre-filter 和 post-filter

- `pre_filter_dense`：先用 distiller 改完整 vocab logits，再让 sampler 做过滤。
  这通常更快，但会改变候选 token 集合，不是最严格的目标语义。
- `post_filter_exact`：LLM 先过滤候选 token，distiller 只修改留下来的候选。
  这是 ESamp distiller intervention 的默认语义。
- `post_filter_dense_cache`：LLM 先过滤候选 token，但 distiller logits 提前按完整
  vocab 算好，候选阶段只 gather。它保留 post-filter 语义，但可能因为完整 vocab
  projection 太贵而不一定更快。

### min-p

min-p 是一种采样过滤规则。它以当前行最大 logit 为参照，只保留相对概率足够高
的 token。等价的 logit-space 判定是：

```text
logit_i >= max_logit + log(min_p)
```

这个形式避免了完整 softmax，是后续优化 exact min-p 路径的关键。

## 有效优化时间线

| 步骤 | 改动 | workload | 吞吐变化 | 为什么有效 |
| --- | --- | --- | --- | --- |
| 1 | 实现第一个 `post_filter_exact` 后端 | 0.5B smoke，`batch=4`，`n=4`，`max_new_tokens=16` | 建立 baseline：`single_off=5400.128`，`model_bank_on=3712.408 tok/s`，ratio `0.6875` | 让目标语义先跑通：LLM 先过滤候选，distiller 只改候选 logits。 |
| 2 | 增加 `pre_filter_dense` 后端 | 同一 smoke workload | ratio `0.6875 -> 0.7088` | 完整 vocab dense logits 避免候选阶段稀疏投影，虽然语义不是最终目标，但提供了性能参考。 |
| 3 | 增加 dense full-row fast path | 同一 smoke workload | ratio `0.7088 -> 0.7461` | 当所有 rows 都受影响时，不再做额外 row-subset 处理，减少 sampler bridge 的 Python/tensor 胶水开销。 |
| 4 | benchmark scratch rows 按 chunk capacity 而不是总 expanded batch 配置 | aligned 0.5B，`effective_batch_cap=64` | init-time shape failure 变成可完成运行 | 这不是直接提速，但让真实 aligned benchmark 能跑起来。 |
| 5 | 清理 CUDA 热路径 CPU sync | 0.5B smoke 和 aligned 0.5B | aligned dense path 后续 `0.6748 -> 0.6998` | 移除 CUDA tensor `tolist()`、CUDA tensor 上的 Python boolean control flow 等隐式同步点。 |
| 6 | 增加 dense-logit precompute cache | 0.5B smoke | best smoke ratio 到 `0.7699` | sampler 消费已经算好的完整 vocab distiller logits，不再在 sampler 阶段重复 dense projection。 |
| 7 | 把 source snapshot copy 放到 precompute stream | 0.5B smoke | ratio `0.7699 -> 0.7714` | 小幅 overlap 收益。copy 不再那么直接压在 sampler/main path 上。 |
| 8 | 缓存 model-bank sampling lookup tensor，把更多 metadata 放到 GPU tensor | 0.5B smoke，两轮平均 | ratio `0.7450 -> 0.7534` | 减少每 step 反复创建 Python/tensor 辅助结构的成本。 |
| 9 | `record_function` profiling scope 改成环境变量开关 | smoke 和 targeted tests | 移除 compile-path warnings，无明显回退 | 默认 benchmark 不再付 profiling context manager 的 Python 开销。 |
| 10 | 加 CUDA event timing，发现 source hook precompute 没进入 cudagraph decode replay | timing smoke | `schedule_attempt_count=1`，`schedule_hit_count=0`，`fallback_count=30` | 找到了核心时序问题：layer hook 里调度的 precompute 在 replayed decode step 中基本没跑到。 |
| 11 | 在 wrapped `model.compute_logits` 处触发 distiller precompute | timing smoke | `fallback_count=30 -> 0`，`wait_ms_avg≈0.0035`；标准 smoke ratio 到 `0.8243` | `compute_logits` 既 graph-safe，又早到足以喂给同一步 sampler。这是第一次稳定越过 79% 节点。 |
| 12 | ESamp step-scope dispatch/train launch 前移，并降低 ESamp stream priority | mid workload，`batch=8`，`n=8`，`max_new_tokens=64` | ratio 约 `0.7421 -> 0.7739` | 更接近真实 decode-heavy workload 时，提前调度 ESamp work 能减少尾部等待和优先级抢占。 |
| 13 | 增加显式 distiller runtime port state | 0.5B smoke | 功能稳定；refactor 后 smoke ratio 约 `0.6854` | 增加 `capture_step_id`、`publish_step_id`、`consume_step_id`，让 publish/consume 状态可观察。这主要是正确性和调试能力。 |
| 14 | 把 `distiller_beta=0` 作为真正 no-op | 0.5B smoke，exact backend | ratio `0.6854 -> 0.8788`；`model_bank_on / single_on = 0.9885` | 数学上 inactive 时不再发布/调度 sampler intervention，把训练开销和 intervention 开销分离。 |
| 15 | 使用 compact precompute cache，并加入 all-random/all-greedy fast paths | 0.5B mid workload | no-op mid ratio 到 `0.8411`；active exact 仍慢 | 同质 sampling batch 不再走 mask/nonzero 混合路径，减少常见路径的分组开销。 |
| 16 | exact random sampling 改为复用已经过滤后的 logits 走 sampler path | 0.5B mid active exact | `~8975 -> ~9020 tok/s` | 正确但收益小，说明 exact random sampling 本身不是主瓶颈。 |
| 17 | model-bank lookup 写入改为 batched `index_copy_` | timing smoke | precompute 平均耗时约 `0.8399 ms -> 0.5937 ms` | 用批量写替代 scalar tensor 写和同步倾向更强的 lookup 准备。吞吐不明显，但 precompute 成本明确下降。 |
| 18 | 增加 `post_filter_dense_cache` 后端 | 0.5B mid active path | 两轮平均 `8667.130 tok/s`，慢于当时 exact/dense 对照 | 证明“保留 post-filter 语义 + dense cache”可行，但完整 vocab projection 每 step 太贵，不自动比 sparse candidate projection 快。 |
| 19 | 增加 tLLM-local `min_p` metadata 和 exact min-p filtering | 0.5B smoke | `model_bank_on=4007.460 tok/s`，`fallback_count=0` | 搭起用户真正关心的 min-p 测量基座。这一步主要是 enablement，不是优化胜利。 |
| 20 | 压缩 7B model-bank 训练 rows，并修复 graph capture | 7B aligned，`beta=0`，`min_p=0.05` | ratio 从约 `79%` 到约 `95.4%`：`single_off≈5047.847`，`model_bank_on≈4817.202 tok/s` | 对 V1 `n>1` prompt expansion，不再训练每个 expanded sample，而是每个 prompt/slot 训练一行；graph capture 也只捕获 active training rows。 |
| 21 | capturable model-bank optimizer 改用 SGD | 7B model-bank graph path | 包含在 `~79% -> ~95.4%` 大跳跃里 | graph capture 不再依赖 AdamW state 初始化，replay 更可靠。 |
| 22 | min-p 阈值改为 logit-space 规则 | active exact min-p path | 避免完整 softmax | `logit_i >= max_logit + log(min_p)` 保持语义，同时移除 full-vocab softmax。 |
| 23 | exact min-p 增加 candidate-internal sampling | 7B active exact min-p | `4305.908 -> 4526.830 tok/s`，约 `85.3% -> 89.7%` | post-filter 修改后不再把完整 masked vocab logits 交回 sampler，而是在候选集合内部采样。 |
| 24 | candidate sampling 复用 min-p keep mask | 0.5B 和 7B active exact min-p | 0.5B 到 `18972.225 tok/s`；7B 避免早先 OOM | 避免第二次创建 `rows x vocab` bool mask，例如 `isfinite(filtered)` 这种完整 vocab mask。 |
| 25 | compact dense-cache retry，并修正 row-map gather | 7B active exact min-p | `4526.830 -> 4566.617 tok/s` | dense cache 改成按 unique prompt rows 计算，并用 `pred_hidden_row_map` 把 expanded candidate rows 映射回 compact rows。 |
| 26 | pure min-p candidate path 去掉 full-vocab clone 和 full-cache cast | 7B active exact min-p | 贡献到最终达标 run | pure min-p 只读 logits 构造 keep mask，不需要 `logits.clone()`；dense-cache 也先 gather 再 cast/move，减少全量内存流量。 |
| 27 | sampler precompute 增加预分配 all-row id buffer | 7B active exact min-p | 贡献到最终达标 run | compact all-row model-bank 路径不再每 decode step 创建新的 `torch.arange(decode_count)`。 |
| 28 | fair 7B min-p target 复测 exact candidate sampling | 7B aligned，active `beta=0.1` | `single_off=4913.897`，`model_bank_on=4792.830`，ratio `0.9754` | 第一次 clean fair result 超过 95%，并且 `loss_count=4080`、model-bank graph replay 正常。 |
| 29 | sampler intervention 重构成 typed sampler-port contracts | 7B fair target after integration | `single_off=4898.855`，`model_bank_on=4663.018`，ratio `0.9519` | 保持目标 ratio 的同时，把通用 tLLM sampler contract 和 ESamp provider 状态解耦。 |
| 30 | 增加 candidate kernel selection counters 和 pre-dispatch fallback | 0.5B 与 7B min-p | 7B 仍高于目标；`candidate_kernel_fallback_count=0` | Triton 不支持 Qwen vocab 时不再每 step 抛异常再 fallback，而是 launch 前直接选择 torch。 |
| 31 | Triton min-p keep-mask 改为显式 opt-in | 7B 两轮 target | `single_off=4800.995`，`model_bank_on=4611.270`，ratio `0.9605` | tiled Triton 正确但收益中性/噪声；默认回 torch 保住已测 7B 目标。 |
| 32 | 增加 ESamp-local model-bank backend boundary 和 `triton_grouped` no-grad backend | 0.5B smoke | `torch=16631.708`，`triton_grouped=16831.864 tok/s` | 证明 Triton grouped model-bank forward 能在真实 runtime 跑，并且不破坏 training graph replay。因证据是单轮且噪声较大，保持 opt-in。 |

## 最重要的经验

1. 第一个大幅收益不是来自新 kernel，而是来自正确调度点。distiller precompute 必须放到
   vLLM decode replay 中实际会执行、且仍然能喂给 sampler 的边界，最终可行点是
   `compute_logits`。

2. CPU sync 是反复出现的敌人。`.tolist()`、CUDA tensor 上的 Python boolean 判断、
   scalar tensor 写入、`active_mask.all()` 这种看似小的操作，在每 step decode 热路径里
   都会被放大。

3. 7B 训练侧达标靠的是减少该训练的东西，而不是把每个 expanded sample 都训练一遍。
   在 V1 `n>1` prompt expansion 下，model-bank 训练应该按 active prompt/slot rows
   进行。

4. 修复 training graph replay 后，active intervention 的主要开销转移到了
   sampler/candidate 路径。candidate-internal min-p sampling 是把 active exact min-p
   从 80% 多推向 95% 的关键一步。

5. full-vocab dense-cache 不自动更快。它能保留 post-filter 语义，但如果每 step 都要做
   完整 vocab projection，可能比 sparse candidate scoring 更慢。

6. Triton 要以目标 workload 结果为准。当前 min-p Triton keep-mask 已测试、已接入，
   但对 Qwen vocab 不是默认收益；torch 默认路径保住了 7B ratio。

## 当前默认策略

- 默认 sampler backend：`post_filter_exact`。
- 默认 model-bank forward backend：`torch`。
- 可选 no-grad model-bank prediction backend：
  `--model-bank-forward-backend triton_grouped`。
- 可选 exact candidate sampling：
  `TLLM_ENABLE_EXACT_CANDIDATE_SAMPLING=1`。
- 可选 Triton min-p keep-mask：
  `TLLM_ENABLE_TRITON_MINP_KEEP_MASK=1`。

稳定目标证据有两组：

- sampler-port cleanup 后的 7B fair min-p 两轮 run：ratio `0.9605`。
- 最终 backend policy cleanup 前的最好 7B fair min-p run：ratio `0.9754`。
