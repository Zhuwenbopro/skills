#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EVAL_AUTOMATION_DIR=${EVAL_AUTOMATION_DIR:-"${SCRIPT_DIR}/../../eval/automation"}
PLAN_FILE=""
CONFIG_FILE="${EVAL_AUTOMATION_DIR}/config.env"
MAX_PARALLEL=0
POLL_INTERVAL=15
MAX_GPUS=8
MAX_GPUS_OVERRIDE=""

usage() {
  cat <<'EOF'
用法：dispatch_plan.sh --plan PLAN.json [选项]

选项：
  --config PATH        auto_eval 共享配置，默认 eval/automation/config.env
  --max-gpus N         只使用编号 0 到 N-1 的 GPU，默认 8
  --max-parallel N     最大并发任务数，0 表示不额外限制
  --poll-interval SEC  调度轮询间隔，默认 15
EOF
}

die() { printf '错误：%s\n' "$*" >&2; exit 2; }
is_nonnegative_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

while (($#)); do
  case "$1" in
    --plan) PLAN_FILE=${2:-}; shift 2 ;;
    --config) CONFIG_FILE=${2:-}; shift 2 ;;
    --max-gpus) MAX_GPUS_OVERRIDE=${2:-}; shift 2 ;;
    --max-parallel) MAX_PARALLEL=${2:-}; shift 2 ;;
    --poll-interval) POLL_INTERVAL=${2:-}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

[[ -n "$PLAN_FILE" && -f "$PLAN_FILE" ]] || die "--plan 指定的计划文件不存在"
[[ -f "$CONFIG_FILE" ]] || die "共享配置不存在：$CONFIG_FILE"
[[ -f "$EVAL_AUTOMATION_DIR/auto_eval.sh" ]] || die "找不到 auto_eval.sh：$EVAL_AUTOMATION_DIR"
is_nonnegative_integer "$MAX_PARALLEL" || die "--max-parallel 必须是非负整数"
is_positive_integer "$POLL_INTERVAL" || die "--poll-interval 必须是正整数"
for cmd in python3 flock rocm-smi; do command -v "$cmd" >/dev/null 2>&1 || die "找不到命令：$cmd"; done

PLAN_FILE=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$PLAN_FILE")
CONFIG_FILE=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CONFIG_FILE")
RUN_ROOT=$(dirname "$PLAN_FILE")
TASK_ROOT="$RUN_ROOT/tasks"
STATUS_FILE="$RUN_ROOT/status.tsv"
DISPATCH_LOG="$RUN_ROOT/dispatcher.log"
mkdir -p "$TASK_ROOT"
exec >>"$DISPATCH_LOG" 2>&1

log() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*"; }

mapfile -t TASK_ROWS < <(python3 - "$PLAN_FILE" <<'PY'
import json, os, sys
from pathlib import Path

plan_path = Path(sys.argv[1])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
seen = set()
seen_results = set()
for item in plan.get("variants", []):
    label = item["label"]
    command = os.path.abspath(os.path.join(plan_path.parent, item["server_command"]))
    result = os.path.abspath(os.path.join(plan_path.parent, item["result_root"]))
    if label in seen:
        raise SystemExit(f"计划含重复 label：{label}")
    if not label or "/" in label or any(c in label for c in "\t\r\n"):
        raise SystemExit(f"计划含非法 label：{label!r}")
    if any(c in command + result for c in "\t\r\n"):
        raise SystemExit("计划路径含非法制表符或换行")
    if result in seen_results:
      raise SystemExit(f"计划含重复 result_root：{result}")
    seen.add(label)
    seen_results.add(result)
    print(f"{label}\t{command}\t{result}")
PY
)
((${#TASK_ROWS[@]} > 0)) || die "计划中没有任务"

# shellcheck source=/dev/null
source "$CONFIG_FILE"
: "${GPU_LOCK_DIR:=/tmp/auto_eval_gpu_locks}"
: "${GPU_VRAM_MAX_PERCENT:=6}"
: "${GPU_HCU_MAX_PERCENT:=0}"
[[ -z "$MAX_GPUS_OVERRIDE" ]] || MAX_GPUS=$MAX_GPUS_OVERRIDE
is_positive_integer "$MAX_GPUS" || die "MAX_GPUS 必须是正整数"
mkdir -p "$GPU_LOCK_DIR"

required_gpus() {
  local command=$1 metadata tp pp
  metadata=$(python3 "$EVAL_AUTOMATION_DIR/lib/server_command_parser.py" metadata "$command") || return 1
  tp=$(sed -n '3p' <<<"$metadata")
  pp=$(sed -n '4p' <<<"$metadata")
  printf '%s\n' "$((tp * pp))"
}

find_available_gpus() {
  local required=$1 output gpu fd selected_csv
  local -a selected=()
  output=$(rocm-smi 2>/dev/null) || return 1
  while IFS= read -r gpu; do
    [[ -n "$gpu" ]] || continue
    exec {fd}>"${GPU_LOCK_DIR}/gpu_${gpu}.lock"
    if flock -n "$fd"; then
      selected+=("$gpu")
      flock -u "$fd"
      exec {fd}>&-
    else
      exec {fd}>&-
    fi
    ((${#selected[@]} >= required)) && break
  done < <(awk -v max_vram="$GPU_VRAM_MAX_PERCENT" -v max_hcu="$GPU_HCU_MAX_PERCENT" \
      -v max_gpus="$MAX_GPUS" '
    $1 ~ /^[0-9]+$/ {
      gpu=$1; vram=$6; hcu=$7; gsub(/%/,"",vram); gsub(/%/,"",hcu)
      if (gpu < max_gpus && (vram+0)<max_vram && (hcu+0)<=max_hcu) print gpu
    }' <<<"$output")
  if ((${#selected[@]} < required)); then
    return 1
  fi
  selected_csv=$(IFS=,; printf '%s' "${selected[*]}")
  GPU_ALLOWLIST=$selected_csv
}

active_count() {
  local count=0 pid
  shopt -s nullglob
  for pid_file in "$TASK_ROOT"/*/worker.pid; do
    pid=$(<"$pid_file")
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then count=$((count + 1)); fi
  done
  printf '%s\n' "$count"
}

write_status() {
  local row label task_dir pid state exit_code worker_pid
  printf 'label\tstate\tlauncher_pid\tworker_pid\texit_code\tlog\n' >"$STATUS_FILE.tmp"
  for row in "${TASK_ROWS[@]}"; do
    IFS=$'\t' read -r label _ _ <<<"$row"
    task_dir="$TASK_ROOT/$label"
    pid=""; state="pending"; exit_code=""; worker_pid=""
    [[ -f "$task_dir/launcher.pid" ]] && pid=$(<"$task_dir/launcher.pid")
    [[ -f "$task_dir/worker.pid" ]] && worker_pid=$(<"$task_dir/worker.pid")
    if [[ -f "$task_dir/launcher.exit" ]]; then
      exit_code=$(<"$task_dir/launcher.exit")
      state=$([[ "$exit_code" == 0 ]] && echo scheduled || echo launch_failed)
    elif [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      state="waiting_for_gpus"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$state" "$pid" "$worker_pid" "$exit_code" "$task_dir/auto_eval.log" >>"$STATUS_FILE.tmp"
  done
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

launch_task() {
  local label=$1 command=$2 result_root=$3 task_dir launcher_pid
  task_dir="$TASK_ROOT/$label"
  mkdir -p "$task_dir"
  (
    set +e
    AUTO_EVAL_CONFIG="$CONFIG_FILE" bash "$EVAL_AUTOMATION_DIR/auto_eval.sh" \
      --server-command "$command" --result-root "$result_root" \
      --gpu-allowlist "$GPU_ALLOWLIST" >"$task_dir/auto_eval.log" 2>&1
    status=$?
    awk -F'PID=' '/评测 worker 已启动：PID=/{split($2,a,"；"); print a[1]; exit}' "$task_dir/auto_eval.log" >"$task_dir/worker.pid"
    printf '%s\n' "$status" >"$task_dir/launcher.exit"
  ) &
  launcher_pid=$!
  printf '%s\n' "$launcher_pid" >"$task_dir/launcher.pid"
  log "已提交 $label，launcher PID=$launcher_pid"
  wait "$launcher_pid"
  [[ "$(<"$task_dir/launcher.exit")" == 0 ]] || die "auto_eval 启动失败：$label"
  [[ -s "$task_dir/worker.pid" ]] || die "auto_eval 未返回 worker PID：$label"
  log "已安排 $label，worker PID=$(<"$task_dir/worker.pid")"
}

log "开始调度 ${#TASK_ROWS[@]} 个对照任务；可用 GPU 编号范围=0-$((MAX_GPUS - 1))"
for row in "${TASK_ROWS[@]}"; do
  IFS=$'\t' read -r label command result_root <<<"$row"
  [[ -f "$command" ]] || die "服务命令不存在：$command"
  need=$(required_gpus "$command") || die "命令校验失败：$label"
  ((need <= MAX_GPUS)) || die "$label 需要 $need 张 GPU，超过 MAX_GPUS=$MAX_GPUS"
  while true; do
    active=$(active_count)
    if ((MAX_PARALLEL == 0 || active < MAX_PARALLEL)) && find_available_gpus "$need"; then break; fi
    log "等待安排 $label：需要 GPU=$need，可用编号范围=0-$((MAX_GPUS - 1))，活跃任务=$active"
    sleep "$POLL_INTERVAL"
  done
  log "为 $label 指定 GPU 候选范围=${GPU_ALLOWLIST}"
  launch_task "$label" "$command" "$result_root"
  write_status
done

write_status
log "所有任务均已提交给 auto_eval；调度器退出"