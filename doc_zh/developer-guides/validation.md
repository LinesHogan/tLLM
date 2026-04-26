# 正确性验证

改完 consumer、runtime 或 producer 后，怎么确认它仍然正确？

核心原则：**不要只跑一种测试。** tLLM 的验证分三层，每层覆盖不同范围，互相不能替代。正确的做法是按改动范围选择对应的层级，全部通过后再提交。

| 验证层级 | 覆盖范围 | 什么时候用 |
|----------|---------|-----------|
| **单元测试** | 纯逻辑、契约、回归 | 任何代码改动后 |
| **GPU correctness** | hidden 对齐、MSE、生成一致性 | 改了 producer、localization 或 runtime hook 后 |
| **功能激活** | training 真的发生、loss 在更新 | 改了 consumer 训练逻辑或参数后 |

下面按场景展开。每个命令都会说明：它在验证什么、参数含义、预期结果、失败时怎么办。

## 普通 consumer 的验证

如果你只改了 consumer 内部逻辑，没碰 runtime 或 producer：

**第一步：跑 DummyConsumer 回归，确认模板本身没坏**

```bash
python -m pytest -q test/test_dummy_consumer_unit.py
python -m pytest -q test/test_consumer_dispatch_contracts_unit.py
```

这两个测试在验证什么：
- `test_dummy_consumer_unit.py`：DummyConsumer 的基本行为——`flows()` 返回正确的 `ConsumerFlow`、`consume_bundle()` 能被调用、`synchronize()` 能排空队列
- `test_consumer_dispatch_contracts_unit.py`：Runtime 和 Consumer 之间的契约——bundle 组装是否正确、port 数据是否按预期传递

预期结果：全部通过（`PASSED`）。

如果失败：说明你可能改到了 consumer 的公共接口或 runtime 的 bundle 组装逻辑。检查 `flows()` 的返回值、`bundle_key` 的设置、以及 consumer 的注册方式。

**第二步：跑你自己的 consumer 单元测试。**

如果你的 consumer 只依赖 public ports，不改 runtime，上面的测试通常足够作为第一轮本地验证。

## 改了 producer 或 runtime

如果你新增 port、改 bundle assembly、改 vLLM hook 时机：

```bash
python -m pytest -q \
  test/test_port_catalog_unit.py \
  test/test_port_bundle_assembler_unit.py \
  test/test_runtime_port_bridge_unit.py
```

这些测试验证什么：
- `test_port_catalog_unit.py`：port 声明是否符合契约。比如你新增了一个 port，它的 locator 语义是否正确
- `test_port_bundle_assembler_unit.py`：bundle 组装逻辑。多个 port 的数据是否能按 `bundle_key` 正确聚合
- `test_runtime_port_bridge_unit.py`：runtime bridge 是否把 producer 的输出正确传递到 consumer

预期结果：全部通过。

如果失败：问题通常出在 port 定义、locator 解析、或 bundle_key 的匹配逻辑。检查你新增/修改的 port 在 `tllm/ports/` 中的定义，以及 `bundle_key` 是否和消费端的 `reads` 声明一致。

## Decode MSE

如果你改了 hidden localization（尤其是 decode 阶段），必须跑 decode MSE。这是验证 producer 是否正确还原了 packed tensor 中每行归属的**最关键测试**。

```bash
python -m verify_v1_decode_rows_minimal \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "hello" \
  --max-new-tokens 8 \
  --mse-tol 1e-4
```

参数说明：

| 参数 | 值 | 为什么这样设 |
|------|-----|-------------|
| `--model-name` | `Qwen/Qwen2.5-0.5B-Instruct` | 小模型，加载快。localization 逻辑和模型大小无关 |
| `--prompt` | `"hello"` | 极简 prompt，减少无关变量 |
| `--max-new-tokens` | `8` | 只生成 8 个 token，快速结束 |
| `--mse-tol` | `1e-4` | 允许的 hidden state 数值误差。如果 localization 逻辑正确，gold 路径和 batched 路径的 hidden 应该几乎完全一致 |

这个验证在做什么：

1. **Gold 路径**：逐个请求单独推理（无 batching）。每个请求有自己的独立 tensor，不存在 packed row 的归属问题
2. **Batched 路径**：多个请求一起跑（vLLM 标准模式）。tensor 是 packed 的，需要 localization 还原每行属于哪个请求

如果 Producer 的 localization 逻辑正确，两条路径提取出的对应 hidden rows 的 MSE 应该小于 `1e-4`。如果 localization 错了（比如行索引算错、phase 判断错了），MSE 会超标。

预期输出：包含 `PASS`。

如果失败：
1. 先确认 `--model-name` 有效且能正常加载
2. 检查 `tllm/producer/decode.py` 中的 localization 逻辑
3. 逐步缩小范围：先跑 decode MSE（最简单），再跑 prefill MSE
4. 将 `--mse-tol` 放宽到 `1e-3`，如果通过说明是数值精度问题；如果仍失败，说明是逻辑错误
5. 跑 `python -m pytest -q test/test_decode_localization_unit.py` 定位具体哪一行 localization 出了问题

## Prefill 验证

Prefill 每个请求可能对应多行，和 decode 的"一请求一行"不同。如果改了 prefill 路径，跑：

```bash
python -m tllm.workflows.repro.repro_prefill_sampling_mse \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-file test/prompt_debug_list.txt \
  --gen-max-new-tokens 4 \
  --sampling-n 3 \
  --mse-tol 1e-5 \
  --gpu-memory-utilization 0.3 \
  --max-model-len 256
```

参数说明：

| 参数 | 值 | 为什么这样设 |
|------|-----|-------------|
| `--model-name` | `Qwen/Qwen2.5-0.5B-Instruct` | 同上，小模型快速验证 |
| `--prompt-file` | `test/prompt_debug_list.txt` | 预设的 prompt 列表，覆盖不同长度和结构 |
| `--gen-max-new-tokens` | `4` | 只生成 4 个 token，重点验证 prefill 阶段 |
| `--sampling-n` | `3` | 每个 prompt 并行采样 3 条，验证多 sample 场景下的 prefill localization |
| `--mse-tol` | `1e-5` | prefill 的 MSE 容忍度比 decode 更严格，因为 prefill 不经过 sampling 随机性 |
| `--gpu-memory-utilization` | `0.3` | prefill 阶段显存占用模式不同，保守分配避免 OOM |
| `--max-model-len` | `256` | 限制序列长度 |

这个验证在做什么：

Prefill 阶段，vLLM 把 prompt tokens 打包成 `[total_prompt_tokens, hidden_size]`。每个请求可能占连续的多行。Producer 需要正确计算每个请求的 `[start, end)` 范围，并提取对应的 hidden rows。这个验证比较 gold 路径和 batched 路径的 prefill hidden，确认范围计算正确。

预期输出：MSE 低于阈值。

如果失败：
1. 检查 `tllm/producer/prefill.py` 中的 `prefill_len` 计算逻辑
2. 确认 `num_scheduled_tokens`、`num_computed_tokens`、`num_prompt_tokens` 的读取是否正确
3. prefill 目前走 eager-first 路径，确认你没有在 graph-safe 路径里改到 prefill 的逻辑

## Sampler / guidance 验证

如果你的 consumer 修改采样或 logits：

```bash
python -m pytest -q \
  test/test_sampler_port_unit.py \
  test/test_sampler_patch_unit.py
```

这两个测试验证什么：
- `test_sampler_port_unit.py`：sampler port 的声明和读取是否正确。比如 candidate logits 的 shape 是否和请求数对齐
- `test_sampler_patch_unit.py`：sampler patch 是否正确安装、是否在正确时机被调用

如果使用 min-p 或候选级 intervention：

```bash
python -m pytest -q test/test_sampler_bridge_minp_unit.py
```

这个测试验证 min-p 过滤后，candidate modifier provider 是否只收到保留的候选，而不是完整词表。

预期结果：全部通过。

如果失败：检查 sampler provider 的注册、`is_active()` 的返回值、以及 provider 返回的 logits delta 的 shape 是否与候选集一致。

## ESamp 正确性

ESamp 训练路径的正确性验证分两步。**不要跳过第二步**——pytest 通过不代表 GPU 训练路径真的在工作。

**第一步：跑单元测试**

```bash
python -m pytest -q \
  test/test_esamp_per_request_unit.py \
  test/test_esamp_distiller_sampling_integration_unit.py \
  test/test_esamp_model_bank_backend_unit.py
```

这些测试验证什么：
- `test_esamp_per_request_unit.py`：per-request 模式的 ESamp 训练机制是否能正确创建参数、执行 forward/backward
- `test_esamp_distiller_sampling_integration_unit.py`：distiller 的预测结果是否能正确接到 sampler bridge
- `test_esamp_model_bank_backend_unit.py`：model-bank 的 slot 分配、参数共享、grouped training 是否正确

**第二步：跑 aligned benchmark，确认 `loss_count > 0`**

```bash
VLLM_USE_FLASHINFER_SAMPLER=1 \
python -m tllm.verification.automated_tests \
  --scenario esamp_loss_parity_qwen2p5_0p5b
```

参数说明：

| 参数 | 值 | 为什么这样设 |
|------|-----|-------------|
| `--scenario` | `esamp_loss_parity_qwen2p5_0p5b` | 预设的验证场景，包含 Qwen2.5-0.5B 模型、标准 workload、三种训练模式的对比 |

这个验证在做什么：

在同一个 aligned workload 下，同时跑 `single_on`、`per_request_on`、`model_bank_on` 三个 case，确认：
1. 三个 case 都报告 `loss_count > 0`（训练确实发生）
2. loss 值和 count 与历史基线保持近似一致（parity check）

为什么要跑这个：pytest 是在隔离环境下测试单个组件，而这个验证是在真实 vLLM 推理流水线上跑端到端训练。很多 bug 只在完整流水线上才会暴露，比如 hook 安装失败、stream 同步错误、CUDA graph capture 问题。

判断标准：

| 检查项 | 合格 | 不合格时怎么办 |
|--------|------|--------------|
| `single_on` 的 `loss_count` | > 0 | 检查 ESamp 训练是否启动、显存是否够 |
| `per_request_on` 的 `loss_count` | > 0 | 检查 per-request 参数分配是否成功 |
| `model_bank_on` 的 `loss_count` | > 0 | 检查 model-bank slot 分配和 flush 逻辑 |
| loss 值与历史基线 | 近似一致 | 如果显著偏离，检查学习率、初始化、或 workload 是否变化 |

如果某个 case 的 `loss_count == 0`：
1. 先检查 consumer 是否拿到了 hidden（`processed_rows > 0`）
2. 检查 deferred launch 是否触发
3. 检查 runtime flow 是否被错误跳过
4. 不要只跑 pytest。pytest 通过不代表 GPU 训练路径真的工作

## 第三方扩展的兼容性清单

提交第三方 consumer 或 runtime 扩展前，确认：

- 不直接依赖 vLLM 私有对象，除非扩展点明确属于 runtime 层
- consumer 面向 `tllm.ports` 和 `PortBundle`
- 异步资源在 `synchronize()` 中可释放
- 热路径没有无意 CPU sync
- 全量 pytest 通过：

```bash
python -m pytest -q
```
