# 开发者测试指南

这篇文档说明：开发 tLLM 时，改完代码后应该跑什么测试、按什么顺序跑。

tLLM 的测试分三层，每层解决不同问题：

| 层级 | 位置 | 解决什么问题 | 运行方式 |
|------|------|-------------|----------|
| **断言型测试** | `test/` | CPU 上的纯逻辑 / 契约 / 回归 | `pytest -q` |
| **Verification harness** | `tllm/verification/` | 需要 GPU 的 pass/fail 验证矩阵 | `python -m tllm.verification...` |
| **手动工作流** | `tllm/workflows/` | benchmark、repro、调参 | `python -m tllm.workflows...` |

规则：
- `test/` 只放 pytest 断言
- `tllm/verification/` 放需要 GPU、带明确 pass/fail 语义的验证
- `tllm/workflows/` 放手动的 benchmark / repro

## TDD 流程

默认采用测试驱动开发：

1. 先写或修改失败测试
2. 运行它，确认真的失败
3. 写最小实现
4. 再跑回归
5. 最后再做必要的重构和文档同步

最低要求：
- 改行为前，必须先有能表达该行为的失败测试
- 不能只靠 benchmark 说"看起来变快了"
- 吞吐提升但 `loss_count == 0` 视为 bug，不视为成功

## 断言型测试

```bash
python -m pytest -q
```

关键测试组：

| 测试 | 关注点 | 什么时候跑 |
|------|--------|-----------|
| `test_decode_localization_unit.py` | decode row 定位是否正确 | 改了 producer 或 localization 逻辑 |
| `test_consumer_dispatch_contracts_unit.py` | PortBundle、ConsumerFlow 契约 | 改了 port 定义或 bundle 组装 |
| `test_dummy_consumer_unit.py` | dummy consumer 最小行为 | 改了 consumer 基类或公共接口 |
| `test_esamp_per_request_unit.py` | ESamp 训练 step 级行为 | 改了 ESamp 训练逻辑 |
| `test_esamp_model_bank_backend_unit.py` | model bank 训练路径 | 改了 model-bank 参数管理 |

最小单测集（只改纯逻辑时）：

```bash
python -m pytest -q \
  test/test_decode_localization_unit.py \
  test/test_consumer_dispatch_contracts_unit.py
```

为什么只跑这两个就够了：decode localization 和 consumer dispatch 是 tLLM 最核心的两条路径。如果它们通过了，大部分 consumer 相关的改动都不会破坏基本契约。

## 改不同模块后该跑什么

### 改 `tllm/producer/` 或 `tllm/runtime/`

Producer 负责从 packed tensor 中提取 hidden rows，runtime 负责装 hook 和 bundle 组装。改了这些，必须先验证 localization 正确：

```bash
python -m pytest -q
python -m verify_v1_decode_rows_minimal \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "hello" \
  --max-new-tokens 8 \
  --mse-tol 1e-4
```

为什么必须跑 MSE：
- pytest 只能验证纯逻辑（比如索引计算是否正确）
- MSE 验证的是在真实 vLLM GPU 流水线上，gold 路径和 batched 路径的 hidden 是否一致
- 很多 localization bug（比如 CUDA graph 下的 buffer 问题）只有在 GPU 上才会暴露

如果 MSE 失败：先读 [正确性验证](../developer-guides/validation.md) 的 Decode MSE 排查指南。

### 改 `tllm/consumers/esamp/` 或训练路径

Consumer 内部的改动通常不影响 localization，但可能影响训练是否正确发生：

```bash
python -m pytest -q
python -m verify_v1_decode_rows_minimal \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "hello" \
  --max-new-tokens 8 \
  --mse-tol 1e-4

# 再补一轮 aligned ESamp benchmark
VLLM_USE_FLASHINFER_SAMPLER=1 \
python -m tllm.workflows.benchmarks.per_request_esamp_benchmark \
  --emit-json-summary \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --benchmark-batch-size 8 \
  --benchmark-max-new-tokens 256 \
  --distiller-lr 1e-3 \
  --model-bank-train-cudagraph \
  --run-model-bank-case
```

为什么 MSE 仍然要跑：改了 consumer 可能意外影响到 runtime 的 hook 时机或 bundle 组装。MSE 是一个快速的全链路回归检查。

benchmark 参数说明：

| 参数 | 值 | 为什么这样设 |
|------|-----|-------------|
| `--benchmark-batch-size` | `8` | 标准 batch size，能测出 batch 效应 |
| `--benchmark-max-new-tokens` | `256` | 够长的 decode 序列，让 ESamp 有充分训练步数 |
| `--distiller-lr` | `1e-3` | 标准学习率 |
| `--model-bank-train-cudagraph` | | 测试 CUDA graph 训练路径是否正常 |
| `--run-model-bank-case` | | 只跑 model-bank 模式，减少验证时间 |

检查重点：

| 检查项 | 合格标准 | 不合格意味着什么 |
|--------|---------|-----------------|
| `single_on` 的 `loss_count` | > 0 | 单模型训练路径有问题 |
| `per_request_on` 的 `loss_count` | > 0 | per-request 参数分配有问题 |
| `model_bank_on` 的 `loss_count` | > 0 | model-bank 的 slot 分配或 flush 有问题 |
| `model_bank_on / single_off` | 在合理区间 | 如果显著低于历史基线，说明新改动引入了额外开销 |

如果要查 ESamp parity 口径，读 [正确性验证](../developer-guides/validation.md) 的 ESamp loss parity guardrail。

## 一句话准则

- 用 `test/` 证明逻辑正确
- 用 `tllm/verification/` 证明 GPU 验证可复跑
- 用 `tllm/workflows/` 做手动 benchmark
- 用 ratio + `loss_count`，而不是单看吞吐，判断 ESamp 是否真的工作
