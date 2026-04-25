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

 当前 capture 存储:
 - decode: `captured_decode[prompt_idx] -> List[Tensor]`
 - prefill: `captured_prefill[prompt_idx] -> List[Tensor]`

 ### Consumer 输入

 Consumer 通过 `ConsumerFlow` 声明需求，通过 `consume_bundle(bundle, ctx)` 接收组装好的 `PortBundle`：

 - decode 本地化后的 hidden（`[rows, hidden_size]`, fp32）
 - 非有效行会被 `decode_valid_mask` 置零
 - 由主 stream 的 ready event 驱动 side stream

 当前支持的读取方式：
 - 读取 `residual_stream` port 获取 source/target hidden
 - 读取 `request_meta` port 获取 request identity
 - 可选在 step 末尾通过 `apply_feedback(ctx)` 执行 delayed backward

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
 5. 在捕获层 hook 中执行:
    - `decode_h1 = scratch.index_select(0, decode_row_idx)`
    - `decode_h1 *= decode_valid_mask`

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
 - `--project unit|decode|prefill|throughput|side_train`：按项目类型过滤
 - `--scenario <scenario_id>`：跑指定场景

 捕获层配置（所有 runner 通用）:
 - `--capture-layer-index <int>`: 解析为 `model.model.layers[idx]`
 - `--capture-layer-path <str>`: 显式路径，例如 `model.model.layers[5]`
