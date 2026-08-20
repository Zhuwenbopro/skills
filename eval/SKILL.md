---
name: eval
description: "Run SGLang EvalScope benchmarks with automatic GPU waiting, GPU locking, port selection, health checks, retries, and dataset-specific generation settings. Use when the user invokes /eval or asks to test a pasted SGLang server command against one or more EvalScope datasets."
argument-hint: "Paste the SGLang launch command, then specify datasets such as math500"
user-invocable: true
---

# SGLang Auto Evaluation

Run SGLang benchmarks through the self-contained [automation](./automation/) interface. Never execute the user's pasted launch command directly.

## Interface Contract

Treat the automation as a black box during normal invocation. The documented interface consists of:

- inputs: `server_command.sh` and the request fields in `config.env`
- validation: `lib/server_command_parser.py metadata`
- entry point: `auto_eval.sh`
- per-run overrides: `auto_eval.sh --server-command PATH --result-root PATH`; command-line values override `config.env`
- status: `/home/eval_results/auto_eval.log`

Do not read `auto_eval.sh`, `eval_command.sh`, the parser implementation, tests, or generated logs before they are needed by the procedure below. Inspect implementation files only when the documented interface fails, behavior contradicts this document, or the user explicitly asks to debug the automation.

## Input

The user may provide:

- A fenced SGLang launch command containing optional `export` or `unset` lines and one `sglang serve` or `python -m sglang.launch_server` command.
- A dataset name or comma-separated dataset list, for example `math500` or `humaneval,math500,gsm8k`.
- An optional request to enable thinking. Thinking is disabled by default.

## Procedure

1. Set `AUTOMATION_DIR` to this Skill's `./automation` directory. Do not search for or copy the automation elsewhere.
2. Extract the complete SGLang command from the user's fenced block and write it to `${AUTOMATION_DIR}/server_command.sh`.
3. Preserve environment assignments, `unset NAME` statements, and SGLang arguments, except for scheduler-owned settings. Normalize the command as follows:
   - remove `nohup`
   - remove trailing `&`
   - remove output redirection such as `>file 2>&1`
   - remove `--port` and its value
   - remove `HIP_VISIBLE_DEVICES` assignments
   - do not add replacement values for the port or GPU visibility; the scheduler selects both
4. Validate the command before starting anything:

   ```bash
   cd "${AUTOMATION_DIR}"
   python3 ./lib/server_command_parser.py metadata ./server_command.sh
   ```

   Stop and report the parser error if the command is invalid.
5. Normalize common dataset spelling for EvalScope. In particular, map `math500` to `math_500`. Do not probe EvalScope or create local test data for standard built-in datasets.
6. Update only these request fields in `${AUTOMATION_DIR}/config.env`:
   - `EVAL_DATASETS` is the normalized comma-separated dataset list.
   - `EVAL_ENABLE_THINKING` is `false` unless the user explicitly asks for `true`.
   - Preserve existing `EVAL_BATCH` and `EVAL_LIMIT` unless the user changes them.
7. Start the scheduler from outside `server_command.sh`. Keep scheduler redirection outside the controlled server command:

   ```bash
   mkdir -p /home/eval_results
   cd "${AUTOMATION_DIR}"
   nohup bash ./auto_eval.sh > /home/eval_results/auto_eval.log 2>&1 &
   echo $!
   ```

8. Start exactly one asynchronous watcher. It must observe both existing and future log lines so that fast startup markers cannot be missed:

   ```bash
    tail -n +1 -F /home/eval_results/auto_eval.log | awk '
       /SGLang 服务健康检查通过/ { health=1 }
       /EvalScope 已启动/ { evalscope=1 }
       health && evalscope { exit 0 }
       /SGLang Server 启动失败|SGLang Server 启动超过|评测期间 SGLang Server 意外退出|EvalScope 评测失败|已达到最大尝试次数|错误：/ { exit 1 }
    '
   ```

   Let terminal completion notification resume the agent. While waiting, do not poll with `grep`, `tail`, `ps`, `get_terminal_output`, or equivalent commands.
9. Handle the watcher result:
   - Success means only that the server passed its health check and EvalScope started. Report the worker PID, result log, datasets, and thinking setting, then return control immediately.
   - Failure means the lifecycle will not retry. Read the scheduler log once, report the first failure marker and its relevant error, then return control.
   - Do not wait for benchmark completion or inspect intermediate reports during a successful initial start.
10. On a later status request, inspect the scheduler log and latest run directory under `RESULT_ROOT`. On a stop request, send `TERM` to the worker PID so it can clean up EvalScope, the SGLang process group, and GPU locks.

## Runtime Behavior

The scheduler waits for free GPUs, locks them, starts one worker, and exits. The worker selects a port, starts SGLang, checks `/health`, starts EvalScope, and cleans up the lifecycle. GPU acquisition may retry indefinitely when `MAX_ATTEMPTS=0`; service or evaluation failure never restarts the lifecycle.

Dataset names are passed directly to EvalScope, which resolves built-in benchmark data. The bundled client supports per-dataset arguments, generation overrides, `--limit`, and grouping datasets that share generation settings.

## Safety and Scope

- Do not run arbitrary shell text from the pasted command outside the controlled `server_command.sh` flow.
- Do not place `nohup`, `&`, fixed ports, fixed GPU visibility, pipes, command substitutions, or log redirection in `server_command.sh`.
- Do not change unrelated project files. Only modify the Skill's `automation/` files for this workflow.
- For the initial run, report that evaluation started only after the scheduler log shows both the SGLang health marker and the EvalScope startup marker. Do not describe that as benchmark completion.
