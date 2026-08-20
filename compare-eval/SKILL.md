---
name: compare-eval
description: '自动执行 SGLang 多参数对照实验或消融实验：根据用户提供的启动命令和环境变量取值生成笛卡尔积 server command，按 GPU allowlist 等待并发调度多个 auto_eval，定期跟踪 EvalScope 数据集评测并汇总准确率与性能指标。Use when: 用户要求 A/B test、参数矩阵、0/1 开关对比、消融实验、不同配置跑同一测试集或比较 benchmark 结果。'
argument-hint: '粘贴 SGLang 命令，指定参数取值、数据集，可选 GPU 范围'
user-invocable: true
---

# SGLang 参数对照评测

把一条基准 SGLang 命令展开成可复现的参数矩阵，复用 `eval` Skill 的 `auto_eval.sh` 完成 GPU 锁、端口、服务健康检查和 EvalScope 生命周期，并汇总各配置结果。

## 必需输入

- 完整的 `sglang serve` 或 `python -m sglang.launch_server` 命令，可带简单的 `export`、变量赋值和 `unset`。
- 一个或多个环境变量及其候选值，例如 `SGLANG_USE_BOLT_RECOMPUTE_W_U=0,1`。
- EvalScope 数据集，例如 `math500`、`humaneval,gsm8k`。

可选输入：thinking、batch、limit、GPU allowlist、最大并发数。用户未提供时，沿用 `eval/automation/config.env`；thinking 默认 `false`。`math500` 规范化为 `math_500`。矩阵默认取笛卡尔积；示例中的两个 0/1 开关产生四组实验。

## 安全边界

1. 不直接执行用户粘贴的命令，只通过生成的命令文件和已有解析器运行。
2. 拒绝命令中的管道、重定向、后台符号、命令替换；移除 `nohup`、尾部 `&`、固定 `--port`、`HIP_VISIBLE_DEVICES` 后再保存模板。
3. 对照项仅接受环境变量名；值 `__UNSET__` 表示该组执行 `unset NAME`。
4. 每次建立唯一目录 `/home/eval_results/comparisons/<timestamp>-<slug>/`，不得覆盖其他实验。
5. 所有变体必须使用相同的数据集、thinking、batch、limit 和基础命令，只有声明的矩阵变量可不同。
6. 提交前展示变体数量；超过 16 组时必须先向用户确认，以免意外占用大量算力。

## 执行流程

1. 定位本 Skill 的 `./automation` 为 `AUTOMATION_DIR`，定位相邻 `../eval/automation` 为 `EVAL_AUTOMATION_DIR`。不要复制或修改 `eval` Skill。
2. 创建唯一 `RUN_ROOT`，保存规范化后的基准命令为 `${RUN_ROOT}/server_command.template.sh`。
3. 运行矩阵生成器。每个参数使用一个独立的 `--matrix`：

   ```bash
   python3 "${AUTOMATION_DIR}/generate_matrix.py" \
     --source "${RUN_ROOT}/server_command.template.sh" \
     --output-dir "${RUN_ROOT}/server_commands" \
     --manifest "${RUN_ROOT}/plan.json" \
     --matrix SGLANG_USE_BOLT_RECOMPUTE_W_U=0,1 \
     --matrix SGLANG_USE_BOLT_MAMBA_STATE_SCATTER=0,1
   ```

4. 对 `plan.json` 中的每个 `server_command` 执行：

   ```bash
   python3 "${EVAL_AUTOMATION_DIR}/lib/server_command_parser.py" metadata <server-command-path>
   ```

   任一失败都停止，不启动部分矩阵。
5. 从 `${EVAL_AUTOMATION_DIR}/config.env` 复制一份 `${RUN_ROOT}/base_config.env`，只修改本次请求字段：`EVAL_DATASETS`、`EVAL_ENABLE_THINKING`、可选的 `EVAL_BATCH`、`EVAL_LIMIT`、`GPU_ALLOWLIST`。不要改共享配置。
6. 启动调度器。它只在足够 GPU 当前可锁时提交下一项；每个 `auto_eval` 仍会原子锁卡，因此并发调度器不会重复占卡。调度器等待所有任务取得 GPU 并得到 worker PID 后退出：

   ```bash
   bash "${AUTOMATION_DIR}/dispatch_matrix.sh" \
     --plan "${RUN_ROOT}/plan.json" \
     --base-config "${RUN_ROOT}/base_config.env" \
     --poll-interval 15
   ```

   用户指定 GPU 时添加 `--gpu-allowlist 0,1,2,3`；指定并发时添加 `--max-parallel N`。必须以异步终端运行该命令，因为等卡可能很久。
7. 不要用固定半小时睡眠，也不要每十分钟主动轮询。让异步终端完成通知恢复 agent；这比睡眠更及时且不浪费调用。调度器退出只表示**所有任务已安排**，不表示评测完成。
8. 调度完成后，启动且只启动一个异步 watcher。它每 60 秒内部检查一次汇总状态，并在所有变体成功或失败后退出；agent 不调用终端输出轮询：

   ```bash
   while true; do
     python3 "${AUTOMATION_DIR}/summarize.py" \
       --plan "${RUN_ROOT}/plan.json" \
       --output "${RUN_ROOT}/summary.json" >/dev/null
     if python3 - "${RUN_ROOT}/summary.json" <<'PY'
   import json, sys
   raise SystemExit(0 if json.load(open(sys.argv[1]))["all_terminal"] else 1)
   PY
     then
       break
     fi
     sleep 60
   done
   ```

   该 watcher 是需要持续运行的异步进程；不要额外 `grep`、`ps`、`tail` 或重复 watcher。
9. watcher 完成后读取 `${RUN_ROOT}/summary.json`。按参数组合列出：状态、各数据集总分和细分 metric、样本数、平均 latency、输出 TPS、失败首行。明确指出最佳分数与性能变化；不把相关性写成因果关系。
10. 若 watcher 因外部原因中断，保留 `RUN_ROOT` 并报告恢复命令。用户要求停止时，只向 `${RUN_ROOT}/tasks/*/worker.pid` 中仍存活的 PID 发送 `TERM`，等待其清理服务和 GPU 锁，不使用宽泛的 `pkill`。

## 输出约定

启动阶段报告：实验目录、矩阵参数、变体数、数据集、GPU 范围。最终报告使用紧凑表格，一行一个变体，并链接/指出 `summary.json`、各任务 `auto_eval.log` 和 EvalScope report。失败变体不阻止汇总其他成功变体，但必须显式标红失败原因，禁止用缺失值冒充 0 分。
