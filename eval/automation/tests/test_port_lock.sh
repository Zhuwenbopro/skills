#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TEST_DIR=$(mktemp -d)
PIDS=()
WORKERS=()

cleanup() {
  local pid
  for pid in "${WORKERS[@]}" "${PIDS[@]}"; do
    [[ -n "$pid" ]] || continue
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  rm -rf -- "$TEST_DIR"
}
trap cleanup EXIT

mkdir -p "$TEST_DIR/bin" "$TEST_DIR/model" "$TEST_DIR/state"

cat >"$TEST_DIR/bin/rocm-smi" <<'EOF'
#!/usr/bin/env bash
echo '0 a b c d 0% 0%'
echo '1 a b c d 0% 0%'
EOF

cat >"$TEST_DIR/bin/curl" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat >"$TEST_DIR/bin/sglang" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$PORT" >"$TEST_STATE/port_${TEST_ID}"
trap 'exit 0' TERM INT
while true; do sleep 1; done
EOF

cat >"$TEST_DIR/eval.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
sleep 3
EOF
chmod +x "$TEST_DIR/bin/rocm-smi" "$TEST_DIR/bin/curl" \
  "$TEST_DIR/bin/sglang" "$TEST_DIR/eval.sh"

cat >"$TEST_DIR/server.sh" <<EOF
sglang serve --model-path $TEST_DIR/model --tp-size 1
EOF

for id in 0 1; do
  cat >"$TEST_DIR/config_${id}.env" <<EOF
HOST=127.0.0.1
HEALTH_HOST=127.0.0.1
SERVER_COMMAND=$TEST_DIR/server.sh
EVAL_COMMAND=$TEST_DIR/eval.sh
RESULT_ROOT=$TEST_DIR/results_${id}
START_PORT=32000
END_PORT=32001
PORT_LOCK_DIR=$TEST_DIR/port_locks
GPU_LOCK_DIR=$TEST_DIR/gpu_locks
GPU_ALLOWLIST=$id
GPU_CONFIRM_SECONDS=0
GPU_POLL_INTERVAL=1
HEALTH_CHECK_INTERVAL=1
SERVER_START_TIMEOUT=5
SHUTDOWN_TIMEOUT=2
FAILURE_LOG_LINES=10
EOF
  PATH="$TEST_DIR/bin:$PATH" TEST_STATE="$TEST_DIR/state" TEST_ID="$id" \
    AUTO_EVAL_CONFIG="$TEST_DIR/config_${id}.env" \
    bash "$PROJECT_DIR/auto_eval.sh" >"$TEST_DIR/auto_eval_${id}.log" 2>&1 &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do wait "$pid"; done
PIDS=()

for id in 0 1; do
  worker=$(awk -F'PID=' '/评测 worker 已启动：PID=/{split($2,a,"；"); print a[1]; exit}' \
    "$TEST_DIR/auto_eval_${id}.log")
  [[ "$worker" =~ ^[0-9]+$ ]]
  WORKERS+=("$worker")
done

deadline=$((SECONDS + 10))
while [[ (! -f "$TEST_DIR/state/port_0" || ! -f "$TEST_DIR/state/port_1") && SECONDS -lt deadline ]]; do
  sleep 1
done
[[ -f "$TEST_DIR/state/port_0" && -f "$TEST_DIR/state/port_1" ]]
port_0=$(<"$TEST_DIR/state/port_0")
port_1=$(<"$TEST_DIR/state/port_1")
[[ "$port_0" != "$port_1" ]]
[[ "$port_0" =~ ^3200[01]$ && "$port_1" =~ ^3200[01]$ ]]

for worker in "${WORKERS[@]}"; do
  deadline=$((SECONDS + 10))
  while kill -0 "$worker" 2>/dev/null && ((SECONDS < deadline)); do sleep 1; done
  if kill -0 "$worker" 2>/dev/null; then
    state=$(awk '{print $3}' "/proc/$worker/stat" 2>/dev/null || true)
    [[ "$state" == Z ]] || { echo "worker 未按时退出：$worker" >&2; exit 1; }
  fi
done
WORKERS=()

echo "PASS: 并发 auto_eval 分配不同端口（$port_0, $port_1）"
