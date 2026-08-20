#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TEST_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEST_DIR"' EXIT

cat >"$TEST_DIR/server.sh" <<'EOF'
export KEEP_ME=hello
export FEATURE_A=9
unset NCCL_TOPO_FILE
python -m sglang.launch_server \
  --model-path /models/example \
  --tp-size 2
EOF

python3 "$SKILL_DIR/automation/generate_matrix.py" \
  --source "$TEST_DIR/server.sh" \
  --output-dir "$TEST_DIR/commands" \
  --manifest "$TEST_DIR/plan.json" \
  --matrix FEATURE_A=0,1 \
  --matrix FEATURE_B=off,on >"$TEST_DIR/count"

[[ "$(<"$TEST_DIR/count")" == 4 ]]
[[ "$(find "$TEST_DIR/commands" -type f | wc -l)" == 4 ]]
python3 - "$TEST_DIR/plan.json" <<'PY'
import json, pathlib, sys
plan = json.load(open(sys.argv[1]))
assert plan["variant_count"] == 4
assert {tuple(v["parameters"].values()) for v in plan["variants"]} == {
    ("0", "off"), ("0", "on"), ("1", "off"), ("1", "on")
}
for variant in plan["variants"]:
    text = pathlib.Path(variant["server_command"]).read_text()
    assert "export KEEP_ME=hello" in text
    assert "unset NCCL_TOPO_FILE" in text
    assert f"export FEATURE_A={variant['parameters']['FEATURE_A']}" in text
    assert f"export FEATURE_B={variant['parameters']['FEATURE_B']}" in text
    assert text.count("FEATURE_A=") == 1
PY

echo 'PASS: 2x2 参数矩阵和命令文件生成正确'
