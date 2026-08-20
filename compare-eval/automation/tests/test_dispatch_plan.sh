#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TEST_DIR=$(mktemp -d)
WORKERS=()

cleanup() {
  local pid
  for pid in "${WORKERS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  rm -rf -- "$TEST_DIR"
}
trap cleanup EXIT

mkdir -p "$TEST_DIR/eval/lib" "$TEST_DIR/bin" "$TEST_DIR/commands" "$TEST_DIR/model"

cat >"$TEST_DIR/bin/rocm-smi" <<'EOF'
#!/usr/bin/env bash
echo '0 a b c d 0% 0%'
echo '1 a b c d 0% 0%'
EOF

cat >"$TEST_DIR/eval/lib/server_command_parser.py" <<'EOF'
#!/usr/bin/env python3
print('/tmp/model')
print('model')
print('1')
print('1')
EOF

cat >"$TEST_DIR/eval/auto_eval.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
label=$(basename "$2" .sh)
printf '%s\n' "$AUTO_EVAL_CONFIG" >"$TEST_STATE/${label}.config"
printf '%s\n' "$@" >"$TEST_STATE/${label}.args"
sleep 30 &
echo "评测 worker 已启动：PID=$!；父进程不再重试"
EOF

chmod +x "$TEST_DIR/bin/rocm-smi" "$TEST_DIR/eval/auto_eval.sh" \
  "$TEST_DIR/eval/lib/server_command_parser.py"
touch "$TEST_DIR/commands/a.sh" "$TEST_DIR/commands/b.sh"
cat >"$TEST_DIR/config.env" <<EOF
GPU_ALLOWLIST=0,1
GPU_LOCK_DIR=$TEST_DIR/gpu_locks
GPU_VRAM_MAX_PERCENT=6
GPU_HCU_MAX_PERCENT=0
EOF
cat >"$TEST_DIR/plan.json" <<'EOF'
{
  "variants": [
    {"label":"a","server_command":"commands/a.sh","result_root":"results/a"},
    {"label":"b","server_command":"commands/b.sh","result_root":"results/b"}
  ]
}
EOF

PATH="$TEST_DIR/bin:$PATH" TEST_STATE="$TEST_DIR" EVAL_AUTOMATION_DIR="$TEST_DIR/eval" \
  bash "$SKILL_DIR/automation/dispatch_plan.sh" \
    --plan "$TEST_DIR/plan.json" --config "$TEST_DIR/config.env" --poll-interval 1

for label in a b; do
  [[ "$(<"$TEST_DIR/${label}.config")" == "$TEST_DIR/config.env" ]]
  grep -Fx -- '--server-command' "$TEST_DIR/${label}.args"
  grep -Fx -- "$TEST_DIR/commands/${label}.sh" "$TEST_DIR/${label}.args"
  grep -Fx -- '--result-root' "$TEST_DIR/${label}.args"
  grep -Fx -- "$TEST_DIR/results/${label}" "$TEST_DIR/${label}.args"
  [[ ! -e "$TEST_DIR/tasks/${label}/config.env" ]]
  WORKERS+=("$(<"$TEST_DIR/tasks/${label}/worker.pid")")
done

echo 'PASS: plan 调度器复用共享配置并传递命令和结果路径'