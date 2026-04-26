# Port 目录

本页是 tLLM **正式数据接口**的完整目录。

## 什么是 Port

Port 是 consumer 向 runtime 声明"我需要什么数据"的方式。Consumer 不直接操作 vLLM 内部对象，而是通过 `ConsumerFlow` 声明自己要读写哪些 port，runtime 负责在正确时机捕获对应数据并组装成 `PortBundle` 传递给 consumer。

当前定义了以下 port 类型：

| Port | 读写类型 | 支持阶段 | 数据来源 |
|------|---------|---------|---------|
| `request_meta` | 只读 | prefill, decode | 请求映射表、request_id、prompt_idx 等 |
| `residual_stream` | 可读可写 | prefill, decode | layer tap 点的 hidden state |
| `cpu_export` | 只写 | prefill, decode | 异步导出到 CPU |
| `logits` | 只读 | decode | 采样前的 logits |
| `kv_cache` | 可读可写 | decode | layer-local KV cache 状态 |
| `token_target` | 只读 | prefill, decode | 训练或 guidance 目标 token |
| `sampler` | 只读 + provider modifier | decode | 采样前 logits、metadata、候选修饰 |

下面逐一展开。

## 当前正式 ports

### `request_meta`

- **类型**：只读
- **支持 phase**：`prefill`、`decode`
- **数据来源**：runtime 中的请求映射表，以及从中派生的 request_id / prompt_idx / sample_idx / phase / step 等元信息

典型用途：

- 给其他 flow 附加 request identity
- 做 source/target 配对
- 做 CPU export 标记

### `residual_stream`

- **类型**：可读、可写
- **支持 phase**：`prefill`、`decode`
- **数据来源**：runtime 在 layer tap 点捕获的 residual hidden 视图，映射为逻辑 `layer + site + phase` locator

当前支持的 site（位置）：

- `block_input`
- `attn_input`
- `attn_output`
- `mlp_input`
- `block_output`

典型用途：

- ESamp adaptive/guidance 路径
- hidden export
- residual delta 写回

### `cpu_export`

- **类型**：只写
- **支持 phase**：`prefill`、`decode`
- **数据来源**：runtime 管理的异步 CPU sink，用于把模型侧数据导出到 CPU，而不把数据库、文件格式或查表逻辑做成公共 port

Locator 字段：

- `channel`
- `format`
- `schema`（可选）

典型用途：

- database consumer
- SAE activation export
- 调试 / 解释性数据导出

### `logits`

- **类型**：只读
- **支持 phase**：`decode`
- **数据来源**：当前 step 采样前的 logits 视图

典型用途：

- classifier guidance
- logits 级分析

### `kv_cache`

- **类型**：可读、可写
- **支持 phase**：`decode`
- **数据来源**：vLLM decode 使用的 layer-local KV cache 状态

Locator 字段：

- `layer`
- `phase`
- `step_scope`

典型用途：

- next-step KV guidance
- KV 可视化或调试

### `token_target`

- **类型**：只读
- **支持 phase**：`prefill`、`decode`
- **数据来源**：当前训练或 guidance 目标对应的 token target

典型用途：

- teacher-forcing 风格目标
- token-level classifier/guidance objective

### `sampler`

- **类型**：只读 port + provider modifier contract
- **支持 phase**：`decode`
- **数据来源**：当前 decode step 的采样前 logits、采样 metadata、对齐的请求元信息，以及可选 source hidden rows

典型用途：

- candidate-level guidance
- same-step sampler intervention
- 在 LLM 的 top-k / top-p / min-p 过滤后，对候选 token logits 做额外修饰

扩展边界：

- tLLM runtime 只负责构造通用 sampler view、安装 sampler patch、调用 `CandidateModifierProvider`
- 具体算法 adapter 放在 consumer 侧，例如 ESamp 的 adapter 位于 `tllm/consumers/esamp/sampler_provider.py`
- generic runtime / port 层不依赖 ESamp consumer，第三方 consumer 可以复用同一 port
