---
name: compare-eval
description: '自动执行 SGLang 多配置对照实验或消融实验：直接生成各实验版本的 server command，按共享配置等待并发调度多个 auto_eval，跟踪 EvalScope 评测并汇总准确率与性能指标。Use when: 用户要求 A/B test、参数对比、0/1 开关对比、消融实验、不同配置跑同一测试集或比较 benchmark 结果。'
argument-hint: '提供基准 SGLang 命令、需要比较的配置、数据集，可选 GPU 范围'
user-invocable: true
---

# SGLang 多配置对照评测

为用户要求的每个实验版本直接生成一份完整的 SGLang 命令，复用 `eval` Skill 的共享配置和 `auto_eval.sh` 生命周期，并汇总所有结果。不要生成参数矩阵、命令模板或任务专属配置。

## 必需输入

- 完整的 `sglang serve` 或 `python -m sglang.launch_server` 基准命令。
- 需要比较的具体版本，例如 baseline、开启某环境变量、增加某 CLI flag。
- EvalScope 数据集，例如 `math500`、`humaneval,gsm8k`。

可选输入：thinking、batch、limit、最大并发数。用户未提供时沿用 `eval/automation/config.env`；thinking 默认 `false`。`math500` 规范化为 `math_500`。

## 安全边界

1. 不直接执行用户粘贴的命令，只通过生成的命令文件和已有解析器运行。
2. 拒绝管道、命令替换及不相关命令；移除 `nohup`、尾部 `&`、重定向、固定 `--port` 和 `HIP_VISIBLE_DEVICES`。
3. 每次建立唯一目录 `/home/eval_results/comparisons/<timestamp>-<slug>/`，不得覆盖其他实验。
4. 各版本必须使用相同的数据集、thinking、batch、limit 和模型配置，仅包含用户要求的配置差异。
5. 启动前向用户列出所有版本；版本数量超过 8 时必须先确认，以免意外占用大量算力。

## 执行流程

1. 定位本 Skill 的 `./automation` 为 `AUTOMATION_DIR`，相邻 `../eval/automation` 为 `EVAL_AUTOMATION_DIR`。
2. 创建唯一 `RUN_ROOT` 及 `${RUN_ROOT}/server_commands/`。
3. Agent 根据实验意图直接写出每个最终命令文件，例如 `001_baseline.sh`、`002_disable_cuda_graph.sh`。命令文件需自包含其环境变量和 CLI 参数，不生成模板或笛卡尔积。
4. 对每个命令执行以下校验；任一失败都停止，不启动部分实验：

   ```bash
   python3 "${EVAL_AUTOMATION_DIR}/lib/server_command_parser.py" metadata <server-command-path>
   ```

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
7. 以异步终端启动调度器。它使用同一份配置调用 `auto_eval.sh --server-command ... --result-root ...`，不会生成任务配置：

   ```bash
   bash "${AUTOMATION_DIR}/dispatch_plan.sh" \
     --plan "${RUN_ROOT}/plan.json" \
     --config "${RUN_ROOT}/config.env" \
     --poll-interval 15
   ```

   未创建本轮配置时，`--config` 指向 `${EVAL_AUTOMATION_DIR}/config.env`。用户指定并发时添加 `--max-parallel N`。GPU 范围由所选共享配置中的 `GPU_ALLOWLIST` 决定。
8. 等调度器的异步完成通知；不要主动轮询。调度器退出只表示所有任务已安排，不表示评测完成。
9. 启动且只启动一个异步 watcher，每 60 秒内部更新 `${RUN_ROOT}/summary.json`，直到 `all_terminal=true`。不要额外轮询日志或启动重复 watcher。
10. watcher 完成后读取 `summary.json`。按版本列出状态、参数说明、各数据集总分和细分 metric、样本数、平均 latency、输出 TPS、失败首行，并指出最佳分数与性能变化；不把相关性写成因果关系。
11. 用户要求停止时，只向 `${RUN_ROOT}/tasks/*/worker.pid` 中仍存活的 PID 发送 `TERM`，等待其释放服务与锁，不使用宽泛的 `pkill`。

## 输出约定

启动阶段报告实验目录、版本清单、数据集和 GPU 范围。最终报告使用紧凑表格，一行一个版本，并指出 `summary.json`、任务日志和 EvalScope report。失败版本不阻止汇总其他成功版本，但必须明确失败原因，禁止用缺失值冒充 0 分。
