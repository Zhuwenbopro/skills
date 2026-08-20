---
name: compare-eval
description: '自动执行 SGLang 多配置对照实验或消融实验：直接生成各实验版本的 server command，按共享配置等待并发调度多个 auto_eval，跟踪 EvalScope 评测并汇总准确率与性能指标。Use when: 用户要求 A/B test、参数对比、0/1 开关对比、消融实验、不同配置跑同一测试集或比较 benchmark 结果。'
argument-hint: '提供基准 SGLang 命令、需要比较的配置、数据集，可选 GPU 范围'
user-invocable: true
---

# SGLang 多配置对照评测

为用户要求的每个实验版本直接生成一份完整的 SGLang 命令，复用 `eval` Skill 的共享配置和 `auto_eval.sh` 生命周期，并汇总所有结果。不要生成参数矩阵、命令模板或任务专属配置。

## 黑盒复用约束

- 将 `eval/automation/auto_eval.sh`、其 `lib/` 和其他内部脚本视为已经验证的黑盒，正常执行本 Skill 时禁止读取、搜索、复制或修改其实现。
- 只使用下方“公开调用契约”给出的命令和参数；不要为了“确认实现”额外检查源码，也不要直接调用 `auto_eval.sh`。
- 允许读取共享 `eval/automation/config.env`，但仅限判断用户请求字段是否需要覆盖，或复制成单份运行配置。
- 调度或评测失败时，先依据 `dispatcher.log`、`tasks/*/auto_eval.log` 和 `summary.json` 报告错误。只有用户明确要求调试底层自动化时，才读取 `auto_eval.sh` 实现。

## 公开调用契约

Agent 只需创建命令文件和 `plan.json`，随后依次调用以下三个入口；无需读取入口源码。

1. 校验一个 server command：

  ```bash
  python3 "${EVAL_AUTOMATION_DIR}/lib/server_command_parser.py" metadata "${SERVER_COMMAND_PATH}"
  ```

  退出码为 `0` 才表示校验成功。每个版本都必须校验，全部成功后才能调度。

2. 异步启动整个计划的调度器：

  ```bash
  bash "${AUTOMATION_DIR}/dispatch_plan.sh" \
    --plan "${RUN_ROOT}/plan.json" \
    --config "${CONFIG_PATH}" \
    --max-gpus "${MAX_GPUS}" \
    --poll-interval 15
  ```

  必填参数是 `--plan`；这里始终显式传入 `--config` 和 `--max-gpus`。可选参数只有 `--max-parallel N` 和 `--poll-interval SEC`。该命令必须使用异步终端运行；不要直接调用各任务的 `auto_eval.sh`。调度器退出表示所有任务已取得 GPU 并启动 worker，不表示评测已完成。

3. 调度器启动后，立即以另一个异步终端启动唯一汇总 watcher：

  ```bash
  bash "${AUTOMATION_DIR}/summary_watcher.sh" \
    --plan "${RUN_ROOT}/plan.json" \
    --output "${RUN_ROOT}/summary.json" \
    --interval 60
  ```

  watcher 会自行定期调用汇总器，并在 `summary.json` 的 `all_terminal` 为 `true` 时退出。不要用 `sleep`、循环读日志或重复启动 watcher 代替它。收到 watcher 完成通知后，只需读取 `summary.json`。

## 必需输入

- 完整的 `sglang serve` 或 `python -m sglang.launch_server` 基准命令。
- 需要比较的具体版本，例如 baseline、开启某环境变量、增加某 CLI flag。
- EvalScope 数据集，例如 `math500`、`humaneval,gsm8k`。

可选输入：thinking、batch、limit、最大并发数、可用 GPU 数。用户未提供时沿用 `eval/automation/config.env`；thinking 默认 `false`。`math500` 规范化为 `math_500`。

## 安全边界

1. 不直接执行用户粘贴的命令，只通过生成的命令文件和已有解析器运行。
2. 拒绝管道、命令替换及不相关命令；移除 `nohup`、尾部 `&`、重定向、固定 `--port` 和 `HIP_VISIBLE_DEVICES`。
3. 每次建立唯一目录 `/home/eval_results/comparisons/<timestamp>-<slug>/`，不得覆盖其他实验。
4. 各版本必须使用相同的数据集、thinking、batch、limit 和模型配置，仅包含用户要求的配置差异。
5. 启动前向用户列出所有版本；版本数量超过 8 时必须先确认，以免意外占用大量算力。

## 执行流程

1. 定位本 Skill 的 `./automation` 为 `AUTOMATION_DIR`，相邻 `../eval/automation` 为 `EVAL_AUTOMATION_DIR`；只拼接公开入口路径，不遍历或读取该目录实现。
2. 创建唯一 `RUN_ROOT` 及 `${RUN_ROOT}/server_commands/`。
3. Agent 根据实验意图直接写出每个最终命令文件，例如 `001_baseline.sh`、`002_disable_cuda_graph.sh`。命令文件需自包含其环境变量和 CLI 参数，不生成模板或笛卡尔积。
4. 按“公开调用契约”校验每个命令；任一失败都停止，不启动部分实验。

5. 仅在本轮确实需要覆盖数据集、thinking、batch 或 limit 时，创建一份 `${RUN_ROOT}/config.env`。它从 `${EVAL_AUTOMATION_DIR}/config.env` 复制并只修改请求字段。否则直接使用共享配置。绝不创建 `base_config.env` 或 `tasks/*/config.env`。
6. Agent 直接写 `${RUN_ROOT}/plan.json`。这是任务索引，不是矩阵；路径可为相对 `plan.json` 的路径：

   ```json
   {
     "variants": [
       {
         "label": "001_baseline",
         "parameters": {"description": "baseline"},
         "server_command": "server_commands/001_baseline.sh",
         "result_root": "results/001_baseline"
       },
       {
         "label": "002_disable_cuda_graph",
         "parameters": {"disable_cuda_graph": true},
         "server_command": "server_commands/002_disable_cuda_graph.sh",
         "result_root": "results/002_disable_cuda_graph"
       }
     ]
   }
   ```

   `label` 必须唯一且不含 `/`；`parameters` 用于最终报告说明差异；每个结果目录必须唯一。
7. 按“公开调用契约”以异步终端启动调度器。`CONFIG_PATH` 是本轮创建的 `${RUN_ROOT}/config.env`；未创建时则是 `${EVAL_AUTOMATION_DIR}/config.env`。`MAX_GPUS` 是用户指定的可用 GPU 数，未指定时为 `8`；用户指定并发时添加 `--max-parallel N`。

  dispatch 从编号 `0` 到 `--max-gpus-1` 中发现一组满足当前空闲标准且锁可用的 GPU，并通过 `auto_eval.sh --gpu-allowlist CSV` 把这组精确候选卡交给任务。`auto_eval` 沿用原有逻辑，只盯住这组卡完成二次状态确认和原子加锁；调度器等待它取得锁并返回 worker PID 后，才为下一个任务寻找另一组卡。`--max-gpus` 默认是 `8`；用户说明本轮只能使用 4 张卡时传 `--max-gpus 4`。不为 GPU 限制生成 config。
8. 调度器启动成功后，立即按“公开调用契约”启动且只启动一个异步 watcher；不要等待调度器退出后才启动 watcher，也不要主动轮询。
9. 等待 watcher 的异步完成通知；不要额外轮询日志或启动重复 watcher。
10. watcher 完成后读取 `summary.json`。按版本列出状态、参数说明、各数据集总分和细分 metric、样本数、平均 latency、输出 TPS、失败首行，并指出最佳分数与性能变化；不把相关性写成因果关系。
11. 用户要求停止时，只向 `${RUN_ROOT}/tasks/*/worker.pid` 中仍存活的 PID 发送 `TERM`，等待其释放服务与锁，不使用宽泛的 `pkill`。

## 输出约定

启动阶段报告实验目录、版本清单、数据集和 GPU 范围。最终报告使用紧凑表格，一行一个版本，并指出 `summary.json`、任务日志和 EvalScope report。失败版本不阻止汇总其他成功版本，但必须明确失败原因，禁止用缺失值冒充 0 分。
