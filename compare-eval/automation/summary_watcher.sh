#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLAN_FILE=""
OUTPUT_FILE=""
INTERVAL=60

usage() {
  cat <<'EOF'
用法：summary_watcher.sh --plan PLAN.json [选项]

选项：
  --output PATH   汇总输出路径，默认 PLAN.json 同目录下的 summary.json
  --interval SEC 更新间隔，默认 60
  -h, --help      显示帮助
EOF
}

die() { printf '错误：%s\n' "$*" >&2; exit 2; }

while (($#)); do
  case "$1" in
    --plan) PLAN_FILE=${2:-}; shift 2 ;;
    --output) OUTPUT_FILE=${2:-}; shift 2 ;;
    --interval) INTERVAL=${2:-}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

[[ -n "$PLAN_FILE" && -f "$PLAN_FILE" ]] || die "--plan 指定的计划文件不存在"
[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || die "--interval 必须是正整数"
PLAN_FILE=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$PLAN_FILE")
[[ -n "$OUTPUT_FILE" ]] || OUTPUT_FILE="$(dirname "$PLAN_FILE")/summary.json"
OUTPUT_FILE=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_FILE")
mkdir -p "$(dirname "$OUTPUT_FILE")"

while true; do
  python3 "$SCRIPT_DIR/summarize.py" --plan "$PLAN_FILE" --output "$OUTPUT_FILE" >/dev/null
  if python3 - "$OUTPUT_FILE" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    raise SystemExit(0 if json.load(f).get("all_terminal") is True else 1)
PY
  then
    exit 0
  fi
  sleep "$INTERVAL"
done
