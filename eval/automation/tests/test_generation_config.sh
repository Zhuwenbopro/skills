#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TEST_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEST_DIR"' EXIT

mkdir -p "$TEST_DIR/bin" "$TEST_DIR/model" "$TEST_DIR/run"

cat >"$TEST_DIR/bin/evalscope" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >"$TEST_STATE/args"
EOF
chmod +x "$TEST_DIR/bin/evalscope"

PATH="$TEST_DIR/bin:$PATH" \
TEST_STATE="$TEST_DIR" \
MODEL_PATH="$TEST_DIR/model" \
PORT=30000 \
RUN_DIR="$TEST_DIR/run" \
EVAL_LOG="$TEST_DIR/eval.log" \
EVAL_DATASETS=humaneval \
EVAL_ENABLE_THINKING=false \
EVAL_BATCH=1 \
EVAL_LIMIT=1 \
  bash "$PROJECT_DIR/eval_command.sh" >"$TEST_DIR/eval.log" 2>&1

generation_config=$(awk '
  previous == "--generation-config" { print; exit }
  { previous = $0 }
' "$TEST_DIR/args")

python3 -c '
import json, sys
config = json.loads(sys.argv[1])
assert config["max_tokens"] == 4096, config
assert config["temperature"] == 0, config
assert config["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False, config
' "$generation_config"

grep -qx -- 'humaneval' "$TEST_DIR/args"
grep -qx -- '1' "$TEST_DIR/args"
echo "generation config test passed"