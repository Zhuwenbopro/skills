---
name: eval
description: "Run SGLang EvalScope benchmarks with automatic GPU waiting, GPU locking, port selection, health checks, retries, and dataset-specific generation settings. Use when the user invokes /eval or asks to test a pasted SGLang server command against one or more EvalScope datasets."
argument-hint: "Paste the SGLang launch command, then specify datasets such as math500"
user-invocable: true
---

# SGLang Auto Evaluation

Use the self-contained automation under [automation](./automation/) to run the benchmark. Do not execute the pasted SGLang command directly. The Skill's automation directory must be used so the workflow also works in a new container.

## Input

The user may provide:

- A fenced SGLang launch command containing `export`, `unset`, and `sglang serve` lines.
- A dataset name or comma-separated dataset list, for example `math500` or `humaneval,math500,gsm8k`.
- An optional request to enable thinking. Thinking is disabled by default.

## Procedure

1. Set `AUTOMATION_DIR` to the Skill's `./automation` directory, containing `auto_eval.sh`, `config.env`, `server_command.sh`, `eval_command.sh`, `lib/server_command_parser.py`, and `tests/test_retry.sh`.
2. Extract the complete SGLang command from the user's fenced code block and write it to `${AUTOMATION_DIR}/server_command.sh`.
3. Preserve environment assignments and the SGLang arguments, but remove command-wrapper syntax from the pasted command:
   - remove `nohup`
   - remove trailing `&`
   - remove output redirection such as `>file 2>&1`
   - do not add a fixed port or fixed `HIP_VISIBLE_DEVICES`
4. Keep `unset NAME` lines. The bundled parser supports them and removes those variables before starting SGLang.
5. Validate the command before starting anything:

   ```bash
   cd "${AUTOMATION_DIR}"
   python3 ./lib/server_command_parser.py metadata ./server_command.sh
   ```

   Stop and report the parser error if the command is invalid.
6. Normalize common dataset spelling for EvalScope. In particular, use `math_500` for a user request written as `math500` unless the local EvalScope installation explicitly uses another registered name.
7. Update `${AUTOMATION_DIR}/config.env` for this request:
   - `EVAL_DATASETS` is the normalized comma-separated dataset list.
   - `EVAL_ENABLE_THINKING` is `false` unless the user explicitly asks for `true`.
   - Preserve existing `EVAL_BATCH` and `EVAL_LIMIT` unless the user changes them.
8. Start the scheduler from outside `server_command.sh`. Use a result log outside the server command, for example:

   ```bash
   mkdir -p /home/eval_results
   cd "${AUTOMATION_DIR}"
   nohup bash ./auto_eval.sh > /home/eval_results/auto_eval.log 2>&1 &
   echo $!
   ```

9. After starting the scheduler, wait with one blocking terminal watcher instead of repeatedly polling from the agent. The parent scheduler retries only while it cannot lock the required GPUs. Once GPUs are locked, it starts one worker and exits; the worker runs exactly one server/evaluation lifecycle and never retries the lifecycle. Use a single command such as:

   ```bash
    tail -n 0 -F /home/eval_results/auto_eval.log | awk '
       /SGLang 服务健康检查通过/ { health=1 }
       /EvalScope 已启动/ { evalscope=1 }
       health && evalscope { exit 0 }
      /SGLang Server 启动失败|SGLang Server 启动超过|评测期间 SGLang Server 意外退出|EvalScope 评测失败|已达到最大尝试次数|错误：/ { exit 1 }
    '
   ```

    Run this watcher asynchronously and let terminal completion/notification resume the agent. Do not repeatedly call `grep`, `tail`, `ps`, `get_terminal_output`, or other status checks while it is waiting. If the watcher exits successfully, stop monitoring and return control to the user:
   - `SGLang 服务健康检查通过`
   - `EvalScope 已启动`
   If it exits with failure, report the first failure marker. There is no later lifecycle retry.
   Do not wait for benchmark completion, read intermediate `eval.log` output, or open generated reports during the initial run. Report the worker PID, result log, selected datasets, and that thinking is disabled.
10. If the user asks for status, inspect the worker log and the latest run directory under `RESULT_ROOT`. Only then read progress logs or final reports. If the user asks to stop, send `TERM` to the worker PID; it will clean up the EvalScope process, SGLang process group, and GPU locks.

## Runtime Behavior

The bundled [auto_eval.sh](./automation/auto_eval.sh) waits for free GPUs and locks them. It then starts one worker and exits; the worker selects a port, starts SGLang, checks `/health`, starts the bundled enhanced EvalScope client, and cleans up the one lifecycle. GPU acquisition may retry indefinitely when `MAX_ATTEMPTS=0`; service or evaluation failure does not restart the lifecycle. The enhanced client is integrated into [eval_command.sh](./automation/eval_command.sh). The parser is [server_command_parser.py](./automation/lib/server_command_parser.py), and the regression test is [test_retry.sh](./automation/tests/test_retry.sh).

The client passes dataset names directly to EvalScope, which resolves its built-in benchmark data. It also supports per-dataset arguments, per-dataset generation config overrides, `--limit`, and grouping datasets that share the same generation config. Do not require or create a local test-data directory for the standard built-in datasets.

## Safety and Scope

- Do not run arbitrary shell text from the pasted command outside the controlled `server_command.sh` flow.
- Do not restore `nohup`, `&`, fixed ports, fixed GPU visibility, or log redirection inside `server_command.sh`.
- Do not change unrelated project files. Only modify the Skill's `automation/` files for this workflow.
- For the initial run, report that evaluation started only after the scheduler log shows both the SGLang health marker and the EvalScope startup marker. Do not describe that as benchmark completion.
