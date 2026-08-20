#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TEST_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEST_DIR"' EXIT
mkdir -p "$TEST_DIR/tasks/a" "$TEST_DIR/results/a/run/reports/model"

cat >"$TEST_DIR/plan.json" <<'EOF'
{"variants":[{"label":"a","parameters":{},"server_command":"a.sh","result_root":"results/a"}]}
EOF
cat >"$TEST_DIR/tasks/a/auto_eval.log" <<'EOF'
EvalScope 评测成功
EOF
cat >"$TEST_DIR/results/a/run/reports/model/math_500.json" <<'EOF'
{"dataset_name":"math_500","score":0.9,"metrics":[],"perf_metrics":{"summary":{}}}
EOF

bash "$SKILL_DIR/automation/summary_watcher.sh" \
  --plan "$TEST_DIR/plan.json" --output "$TEST_DIR/summary.json" --interval 1

python3 - "$TEST_DIR/summary.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    summary = json.load(f)
assert summary["all_terminal"] is True
assert summary["variants"][0]["state"] == "success"
PY

echo 'PASS: summary watcher 在全部任务终态后退出'