# vLLM 兼容性

这篇文档说明 tLLM 如何处理不同 vLLM 版本的差异。

tLLM 通过 monkey-patch 在 vLLM `GPUModelRunner` 的关键生命周期里安装 hook。vLLM 在不同版本之间会改变内部接口，尤其是 `_prepare_inputs` 的返回结构。为了兼容多个版本，tLLM 引入了一个**版本化的 adapter 层**，把不同版本的接口归一化，然后 runtime 代码再消费统一后的视图。

这篇文档覆盖：
1. tLLM 支持哪些 vLLM 版本
2. Adapter 归一化后的视图包含什么
3. 不同版本的 tuple layout 怎么解析
4. 新增 vLLM 版本时该怎么扩展

## 支持的版本

- `0.7.x`
- `0.10.x`
- `0.11+`

当前开发环境锁定在 `vllm==0.10.1.1`，所以 `0.10.x` 是日常验证的主要目标。

`0.7.x` 主要作为历史兼容底线和旧实验日志的参考。

## Adapter 做了什么

vLLM 的 `_prepare_inputs` 返回结构在不同版本里会变。tLLM 通过一个版本化的 adapter 把它归一化，然后 runtime 代码再消费。

归一化后的视图包含：

- `attn_metadata`
- `logits_indices`
- `spec_decode_common`
- `num_scheduled_tokens_np`

## 当前解析规则

- 如果 `_prepare_inputs` 返回 6-tuple：
  - `attn_metadata = out[0]`
  - `logits_indices = out[1]`
  - `num_scheduled_tokens_np = out[3]`
  - `spec_decode_common = out[4]`
- 如果 `_prepare_inputs` 返回 2-tuple：
  - `attn_metadata = out[0]`
  - `logits_indices = out[1]`
  - `spec_decode_common = None`
  - `num_scheduled_tokens_np` 从 `scheduler_output.num_scheduled_tokens` 推导

## Adapter 家族

- `V072PrepareInputsAdapter` 处理 `0.7.x`
- `V010PrepareInputsAdapter` 处理 `0.10.x`
- `V011PlusPrepareInputsAdapter` 处理 `0.11+`

当前实现共享一个解析核心，但按版本分开保持了边界清晰。如果未来 vLLM 再次改变 tuple layout，有稳定的地方做特化。

## 哪些代码在用

- `verify_v1_decode_rows_minimal.py`（开发者 correctness 验证入口）
- `tllm/workflows/repro/prefill_capture_support.py`
- `tllm/runtime/residual_runtime.py`

以上入口先用 adapter 家族归一化 vLLM tuple layout，然后再做 localization 或验证。

- `tllm/runtime/vllm_patch/port_runtime_hooks.py`
- `tllm/runtime/ports/`

以上代码在 localize decode rows 和组装 residual/runtime bundle 时，也使用同一个归一化后的 adapter 视图。
