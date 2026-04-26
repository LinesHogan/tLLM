# Producer/Consumer 契约

这篇文档描述 tLLM 中 Producer 和 Consumer 之间的**数据契约**。它回答的核心问题是：Producer 从 vLLM 中提取了什么数据、Consumer 收到的是什么样的数据、两者之间的约定是什么。

 Producer 的职责是从 vLLM 的 packed tensor 中定位并提取正确的 hidden rows。Consumer 的职责是拿到这些 rows 后做分析、训练或反馈。Runtime 在两者之间做桥接：装 hook、维护定位信息、把数据聚合成 bundle。

 这篇文档覆盖三个层面：
 1. **数据格式** —— Producer 输出什么、Consumer 输入什么
 2. **交互方式** —— Port-based 消费的具体约定
 3. **核心算法** —— decode 和 prefill 的 localization 怎么做

 ## 数据格式

 ### Producer 输出

 Producer 从 vLLM 的 packed tensor 中提取以下数据：

 - `hidden`: 捕获层的 hidden 选中行（默认第一层，可配置）
 - 元信息:
   - `phase`: decode / prefill
   - `prompt_idx`
   - `sample_idx`（支持 `n>1`）
   - prefill 时可附带 token offset

 当前 decode capture 使用按 resolved layer path 索引的 runtime tap buffer：
 - `tap_decode_hidden[resolved_path] -> Tensor[rows, hidden_size]`

 hook 会把本 step 的 decode rows 写入这些固定 buffer。旧的 producer helper 或 prefill
 repro workflow 可能仍然使用按 prompt index 组织的存储，但现代 consumer delivery 应该
 通过 port 和 bundle 理解，而不是依赖那种内部存储形状。

 ### Consumer 输入

 Consumer 通过 `ConsumerFlow` 声明需求，通过 `consume_bundle(bundle, ctx)` 接收组装好的 `PortBundle`：

 - decode 本地化后的 hidden（`[rows, hidden_size]`，dtype 跟捕获层一致）
 - runtime 投递本 step 的 active rows；hook 内部仍使用固定 scratch buffer 以兼容 CUDA Graph replay
 - 普通 bundle 的 tensor 按 borrowed view 理解，device lease 的生命周期见下文

 当前支持的读取方式：
 - 读取 `residual_stream` port 获取 source/target hidden
 - 读取 `request_meta` port 获取 request identity
 - 可选在 step 末尾通过 `on_step_end(ctx)` 执行 delayed backward

 `ConsumerFlow` 的默认投递元数据是 `delivery="bundle"` 和
 `ownership="borrowed"`，对应标准的 bundle 分发路径。若 Consumer 已准备好
 直接消费 runtime 租借的 device tensor，可显式选择
 `delivery="device_lease"` 与 `ownership="runtime_lease"`。ESamp 在这里应理解为
 一类 adaptive/guidance consumer，而不是 ESamp 训练机制本身的同义词。

 当前实现里的 device tensor lease 描述的是 runtime-owned tensor entries 和 active row
 count。这些 entries 只保证在本次 `consume_bundle()` 调用期间有效，并且必须按只读
 数据使用；lease 会用 `lifetime="consume_call"` 明确表达这一点。lease 暂不携带
 ready events，也不承诺 durable-buffer 生命周期；如果 consumer 需要在调用结束后继续
 持有数据，必须先拷贝到自己的 buffer。

 当前 `device_lease` 投递只覆盖 `bundle_key=("engine_step_id", "phase")` 的 decode
 step bundle；读取类型支持 `residual_stream` 和可选的 `request_meta`。更广的 port
 覆盖和 durable staged-buffer lease 应该作为新的 contract revision 加入，而不是从
 当前实现中推断出来。

 Flow 还可以用 `row_compaction` 声明行形状。默认是 `row_compaction="none"`，
 保留 decode row 的顺序和数量。`row_compaction="first_per_prompt"` 表示 runtime
 只给当前 decode step 中每个 prompt 的第一行。当 request metadata 以
 `RowBatchMeta` 投递时，`row_ids` 记录这些行原本的 decode-row 位置。metadata 的行数
 匹配投递的 live rows。这是面向 per-prompt GPU consumer 的通用投递契约；ESamp
 model-bank 会使用它，但 runtime 不会特判 ESamp。

 关于普通 bundle 路径和 device-lease 路径的教学式对比，见
 [Consumer 投递模式](../developer-guides/consumer-delivery-modes.md)。

 ## 核心算法

 ### Decode localization（graph-safe）

 输入:
 - `req_ids`
 - `is_decode_req`
 - `logits_indices`
 - `num_actual_tokens`

 步骤:
 1. `decode_positions = [i for i in req_ids if is_decode_req[i]]`
 2. `row_idx = logits_indices[decode_positions]`
 3. 写入固定 buffer: `decode_row_idx`
 4. 写 `decode_valid_mask[:k] = 1`
 5. 在捕获层 hook 中按固定 buffer shape 执行 graph-safe gather
 6. 用 `decode_valid_mask` 清掉非 active rows

 ### Prefill localization（eager-first）

 每个 request:
 - `scheduled = num_scheduled_tokens[r]`
 - `computed = num_computed_tokens[r]`
 - `prompt_len = num_prompt_tokens[r]`
 - `prefill_len = clamp(prompt_len - computed, 0, scheduled)`

 若当前 request packed 区间是 `[row_base, row_base + scheduled)`，则 prefill 行为 `[row_base, row_base + prefill_len)`。

 ## 运行与验证

 下面两个命令用来验证 Producer/Consumer 契约是否正确履行。

 ### Prefill teacher-forcing MSE

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

 这个命令验证 prefill 阶段的 localization 是否正确。Prefill 中每个请求可能对应多行，producer 需要正确计算每个请求的 `[start, end)` 范围。验证原理和 decode MSE 相同：比较 gold 路径和 batched 路径的 hidden rows。

 参数说明见 [正确性验证](../developer-guides/validation.md) 的 Prefill 验证章节。

 可单独跑 phase:
 - n=1: `--run-phase-a --no-run-phase-b`
 - n>1: `--no-run-phase-a --run-phase-b`

 这两个 flag 控制是否跑 prefill 的两个子阶段。`phase-a` 是 n=1 的 prefill，`phase-b` 是 n>1 的 prefill。分别验证可以缩小问题范围。

 ### 自动回归验证矩阵

 ```bash
 python -m tllm.verification.automated_tests \
   --list
 ```

 这个命令列出所有可用的自动化验证场景，不实际执行。用来查看当前有哪些预定义的验证矩阵。

 实际跑某个场景：
 ```bash
 python -m tllm.verification.automated_tests \
   --scenario esamp_loss_parity_qwen2p5_0p5b
 ```

 过滤方式:
 - `--project unit|decode|prefill|throughput|esamp`：按项目类型过滤
 - `--scenario <scenario_id>`：跑指定场景

 捕获层配置（所有 runner 通用）:
 - `--capture-layer-index <int>`: 解析为 `model.model.layers[idx]`
 - `--capture-layer-path <str>`: 显式路径，例如 `model.model.layers[5]`
