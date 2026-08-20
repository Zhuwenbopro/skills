#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TEST_DIR=$(mktemp -d)
RUNNER_PID=""

cleanup() {
  if [[ -n "$RUNNER_PID" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
    kill -TERM "$RUNNER_PID" 2>/dev/null || true
    wait "$RUNNER_PID" 2>/dev/null || true
  fi
  rm -rf -- "$TEST_DIR"
}
trap cleanup EXIT

mkdir -p "$TEST_DIR/bin" "$TEST_DIR/model" "$TEST_DIR/state"

cat >"$TEST_DIR/bin/rocm-smi" <<'EOF'
#!/usr/bin/env bash
echo '0 a b c d 0% 0%'
EOF

cat >"$TEST_DIR/bin/curl" <<'EOF'
#!/usr/bin/env bash
[[ -f "$TEST_STATE/healthy" ]]
EOF

cat >"$TEST_DIR/bin/sglang" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == "serve" ]]
printf '%s\n' "$@" >"$TEST_STATE/server_args"
printf '%s\n' "$HIP_VISIBLE_DEVICES" >"$TEST_STATE/server_gpus"
count_file="$TEST_STATE/server_count"
count=0
[[ -f "$count_file" ]] && count=$(<"$count_file")
count=$((count + 1))
echo "$count" >"$count_file"
touch "$TEST_STATE/healthy"
trap 'rm -f "$TEST_STATE/healthy"; exit 0' TERM INT
while true; do sleep 1; done
EOF

cat >"$TEST_DIR/server_command.sh" <<EOF
export TEST_PASTED_ENV=enabled
export HIP_VISIBLE_DEVICES=6,7
sglang serve --model-path $TEST_DIR/model
--port 39999
--tp-size=1
--pp-size 1
--trust-remote-code
EOF

cat >"$TEST_DIR/eval.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "${EVAL_DATASETS:-}" == "humaneval" ]]
[[ "${EVAL_ENABLE_THINKING:-}" == "false" ]]
[[ "${EVAL_BATCH:-}" == "7" ]]
[[ "${EVAL_LIMIT:-}" == "None" ]]
count_file="$TEST_STATE/eval_count"
count=0
[[ -f "$count_file" ]] && count=$(<"$count_file")
count=$((count + 1))
echo "$count" >"$count_file"
echo '模拟评测失败'
exit 9
EOF

chmod +x "$TEST_DIR/bin/rocm-smi" "$TEST_DIR/bin/curl" \
  "$TEST_DIR/bin/sglang" "$TEST_DIR/eval.sh"

cat >"$TEST_DIR/config.env" <<EOF
HOST="127.0.0.1"
HEALTH_HOST="127.0.0.1"
SERVER_COMMAND="$TEST_DIR/server_command.sh"
EVAL_COMMAND="$TEST_DIR/eval.sh"
EVAL_ENABLE_THINKING=false
EVAL_DATASETS=humaneval
EVAL_BATCH=7
EVAL_LIMIT=None
RESULT_ROOT="$TEST_DIR/results"
START_PORT=31000
END_PORT=31010
SERVER_START_TIMEOUT=5
HEALTH_CHECK_INTERVAL=1
GPU_VRAM_MAX_PERCENT=5
GPU_HCU_MAX_PERCENT=0
GPU_POLL_INTERVAL=1
GPU_CONFIRM_SECONDS=0
GPU_ALLOWLIST=""
MAX_ATTEMPTS=4
SHUTDOWN_TIMEOUT=2
FAILURE_LOG_LINES=10
GPU_LOCK_DIR="$TEST_DIR/locks"
EOF

PATH="$TEST_DIR/bin:$PATH" \
TEST_STATE="$TEST_DIR/state" \
AUTO_EVAL_CONFIG="$TEST_DIR/config.env" \
  bash "$PROJECT_DIR/auto_eval.sh" >"$TEST_DIR/runner.log" 2>&1 &
RUNNER_PID=$!

deadline=$((SECONDS + 15))
while kill -0 "$RUNNER_PID" 2>/dev/null && ((SECONDS < deadline)); do
  sleep 1
done

if kill -0 "$RUNNER_PID" 2>/dev/null; then
  echo '测试超时' >&2
  cat "$TEST_DIR/runner.log" >&2
  exit 1
fi

if wait "$RUNNER_PID"; then
  status=0
else
  status=$?
fi
RUNNER_PID=""

if [[ "$status" -ne 0 ]]; then
  echo "等待器异常退出，状态=$status" >&2
  cat "$TEST_DIR/runner.log" >&2
  exit 1
fi
deadline=$((SECONDS + 15))
while [[ ! -f "$TEST_DIR/state/eval_count" ]] && ((SECONDS < deadline)); do
  sleep 1
done
[[ "$(<"$TEST_DIR/state/server_count")" -eq 1 ]]
[[ "$(<"$TEST_DIR/state/eval_count")" -eq 1 ]]
[[ "$(<"$TEST_DIR/state/server_gpus")" == "0" ]]
grep -qx -- '31000' "$TEST_DIR/state/server_args"
grep -qx -- 'model' "$TEST_DIR/state/server_args"
grep -q '启动本轮 worker 后父进程退出' "$TEST_DIR/runner.log"
grep -q '父进程不再重试' "$TEST_DIR/runner.log"
deadline=$((SECONDS + 15))
while [[ -f "$TEST_DIR/state/healthy" ]] && ((SECONDS < deadline)); do
  sleep 1
done
[[ ! -f "$TEST_DIR/state/healthy" ]]

echo 'PASS: 参数解析、自动端口/GPU 覆盖、单次 worker 执行和失败清理均正常'
