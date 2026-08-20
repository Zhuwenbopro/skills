#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TEST_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEST_DIR"' EXIT
mkdir -p "$TEST_DIR/tasks/a" "$TEST_DIR/tasks/b" \
  "$TEST_DIR/results/a/run/reports/model"

cat >"$TEST_DIR/plan.json" <<EOF
{
  "variant_count": 2,
  "variants": [
    {"label": "a", "parameters": {"FEATURE": "0"}, "server_command": "$TEST_DIR/a.sh", "result_root": "results/a"},
    {"label": "b", "parameters": {"FEATURE": "1"}, "server_command": "$TEST_DIR/b.sh", "result_root": "results/b"}
  ]
}
EOF
cat >"$TEST_DIR/tasks/a/auto_eval.log" <<'EOF'
EvalScope 已启动
EvalScope 评测成功
EOF
cat >"$TEST_DIR/tasks/b/auto_eval.log" <<'EOF'
SGLang Server 已启动
EvalScope 已启动
EOF
cat >"$TEST_DIR/results/a/run/reports/model/math_500.json" <<'EOF'
{"dataset_name":"math_500","score":0.8,"metrics":[{"name":"acc","score":0.8}],"perf_metrics":{"summary":{"n_samples":10}}}
EOF

python3 "$SKILL_DIR/automation/summarize.py" --plan "$TEST_DIR/plan.json" \
  --output "$TEST_DIR/summary.json" >/dev/null
python3 - "$TEST_DIR/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
assert summary["all_terminal"] is False
assert summary["variants"][0]["state"] == "success"
assert summary["variants"][0]["metrics"][0]["score"] == 0.8
assert summary["variants"][1]["state"] == "running"
PY

echo 'PASS: 运行状态和 EvalScope 指标汇总正确'
