#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EVAL_AUTOMATION_DIR=${EVAL_AUTOMATION_DIR:-"${SCRIPT_DIR}/../../eval/automation"}
PLAN_FILE=""
BASE_CONFIG="${EVAL_AUTOMATION_DIR}/config.env"
GPU_ALLOWLIST_OVERRIDE=""
MAX_PARALLEL=0
POLL_INTERVAL=15

usage() {
  cat <<'EOF'
用法：dispatch_matrix.sh --plan PLAN.json [选项]

选项：
  --base-config PATH       auto_eval 基础配置，默认 eval/automation/config.env
  --gpu-allowlist CSV      覆盖基础配置中的 GPU_ALLOWLIST
  --max-parallel N         最大并发任务数，0 表示不额外限制
  --poll-interval SEC      调度轮询间隔，默认 15
EOF
}

die() { printf '错误：%s\n' "$*" >&2; exit 2; }
is_nonnegative_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

while (($#)); do
  case "$1" in
    --plan) PLAN_FILE=${2:-}; shift 2 ;;
    --base-config) BASE_CONFIG=${2:-}; shift 2 ;;
    --gpu-allowlist) GPU_ALLOWLIST_OVERRIDE=${2:-}; shift 2 ;;
    --max-parallel) MAX_PARALLEL=${2:-}; shift 2 ;;
    --poll-interval) POLL_INTERVAL=${2:-}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

[[ -n "$PLAN_FILE" && -f "$PLAN_FILE" ]] || die "--plan 指定的计划文件不存在"
[[ -f "$BASE_CONFIG" ]] || die "基础配置不存在：$BASE_CONFIG"
[[ -x "$EVAL_AUTOMATION_DIR/auto_eval.sh" || -f "$EVAL_AUTOMATION_DIR/auto_eval.sh" ]] || \
  die "找不到 auto_eval.sh：$EVAL_AUTOMATION_DIR"
is_nonnegative_integer "$MAX_PARALLEL" || die "--max-parallel 必须是非负整数"
is_positive_integer "$POLL_INTERVAL" || die "--poll-interval 必须是正整数"
for cmd in python3 flock; do command -v "$cmd" >/dev/null 2>&1 || die "找不到命令：$cmd"; done

PLAN_FILE=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$PLAN_FILE")
BASE_CONFIG=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$BASE_CONFIG")
RUN_ROOT=$(dirname "$PLAN_FILE")
TASK_ROOT="$RUN_ROOT/tasks"
STATUS_FILE="$RUN_ROOT/status.tsv"
DISPATCH_LOG="$RUN_ROOT/dispatcher.log"
mkdir -p "$TASK_ROOT"
exec >>"$DISPATCH_LOG" 2>&1

log() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*"; }

mapfile -t TASK_ROWS < <(python3 - "$PLAN_FILE" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
for item in plan.get("variants", []):
    label = item["label"]
    command = item["server_command"]
    if any(char in label for char in "\t\r\n") or any(char in command for char in "\t\r\n"):
        raise SystemExit("计划含非法制表符或换行")
    print(f"{label}\t{command}")
PY
)
((${#TASK_ROWS[@]} > 0)) || die "计划中没有任务"

# shellcheck source=/dev/null
source "$BASE_CONFIG"
: "${RESULT_ROOT:=/home/eval_results}"
: "${GPU_LOCK_DIR:=/tmp/auto_eval_gpu_locks}"
: "${GPU_ALLOWLIST:=}"
: "${EVAL_DATASETS:=math_500}"
: "${EVAL_ENABLE_THINKING:=false}"
: "${EVAL_BATCH:=64}"
: "${EVAL_LIMIT:=None}"
: "${EVAL_COMMAND:=${EVAL_AUTOMATION_DIR}/eval_command.sh}"
[[ -n "$GPU_ALLOWLIST_OVERRIDE" ]] && GPU_ALLOWLIST=$GPU_ALLOWLIST_OVERRIDE
GPU_ALLOWLIST=${GPU_ALLOWLIST//[[:space:]]/}
[[ -z "$GPU_ALLOWLIST" || "$GPU_ALLOWLIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "GPU allowlist 格式错误"

PLAN_RESULT_ROOT="$RUN_ROOT/results"
mkdir -p "$PLAN_RESULT_ROOT" "$GPU_LOCK_DIR"

variant_field() {
  python3 - "$PLAN_FILE" "$1" "$2" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
label, field = sys.argv[2:]
item = next(v for v in plan["variants"] if v["label"] == label)
print(item[field])
PY
}

required_gpus() {
  local command=$1 metadata tp pp
  metadata=$(python3 "$EVAL_AUTOMATION_DIR/lib/server_command_parser.py" metadata "$command") || return 1
  tp=$(sed -n '3p' <<<"$metadata")
  pp=$(sed -n '4p' <<<"$metadata")
  printf '%s\n' "$((tp * pp))"
}

count_free_allowed_gpus() {
  local output
  output=$(rocm-smi 2>/dev/null) || return 1
  awk -v max_vram="${GPU_VRAM_MAX_PERCENT:-6}" -v max_hcu="${GPU_HCU_MAX_PERCENT:-0}" \
      -v allowed="$GPU_ALLOWLIST" '
    BEGIN { n=split(allowed,a,","); for(i=1;i<=n;i++) ok[a[i]]=1 }
    $1 ~ /^[0-9]+$/ {
      gpu=$1; vram=$6; hcu=$7; gsub(/%/,"",vram); gsub(/%/,"",hcu)
      if ((allowed=="" || gpu in ok) && (vram+0)<max_vram && (hcu+0)<=max_hcu) print gpu
    }' <<<"$output" | while IFS= read -r gpu; do
      exec {fd}>"${GPU_LOCK_DIR}/gpu_${gpu}.lock"
      if flock -n "$fd"; then echo "$gpu"; flock -u "$fd"; fi
      exec {fd}>&-
    done | wc -l
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
  local row label task_dir pid state exit_code="" worker_pid=""
  printf 'label\tstate\tlauncher_pid\tworker_pid\texit_code\tlog\n' >"$STATUS_FILE.tmp"
  for row in "${TASK_ROWS[@]}"; do
    IFS=$'\t' read -r label _ <<<"$row"
    task_dir="$TASK_ROOT/$label"
    pid=""; state="pending"
    [[ -f "$task_dir/launcher.pid" ]] && pid=$(<"$task_dir/launcher.pid")
    [[ -f "$task_dir/worker.pid" ]] && worker_pid=$(<"$task_dir/worker.pid") || worker_pid=""
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
  local label=$1 command=$2 task_dir config launcher_pid
  task_dir="$TASK_ROOT/$label"
  mkdir -p "$task_dir"
  config="$task_dir/config.env"
  cp "$BASE_CONFIG" "$config"
  cat >>"$config" <<EOF
SERVER_COMMAND=$(printf '%q' "$command")
EVAL_COMMAND=$(printf '%q' "$EVAL_COMMAND")
RESULT_ROOT=$(printf '%q' "$PLAN_RESULT_ROOT/$label")
EVAL_DATASETS=$(printf '%q' "$EVAL_DATASETS")
EVAL_ENABLE_THINKING=$(printf '%q' "$EVAL_ENABLE_THINKING")
EVAL_BATCH=$(printf '%q' "$EVAL_BATCH")
EVAL_LIMIT=$(printf '%q' "$EVAL_LIMIT")
GPU_ALLOWLIST=$(printf '%q' "$GPU_ALLOWLIST")
GPU_LOCK_DIR=$(printf '%q' "$GPU_LOCK_DIR")
EOF
  (
    set +e
    AUTO_EVAL_CONFIG="$config" bash "$EVAL_AUTOMATION_DIR/auto_eval.sh" >"$task_dir/auto_eval.log" 2>&1
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

log "开始调度 ${#TASK_ROWS[@]} 个对照任务；GPU_ALLOWLIST=${GPU_ALLOWLIST:-全部}"
for row in "${TASK_ROWS[@]}"; do
  IFS=$'\t' read -r label command <<<"$row"
  need=$(required_gpus "$command") || die "命令校验失败：$label"
  while true; do
    active=$(active_count)
    free=$(count_free_allowed_gpus || echo 0)
    if (( (MAX_PARALLEL == 0 || active < MAX_PARALLEL) && free >= need )); then break; fi
    log "等待安排 $label：需要 GPU=$need，当前可锁空闲 GPU=$free，并发等待器=$active"
    sleep "$POLL_INTERVAL"
  done
  launch_task "$label" "$command"
  write_status
 done

write_status
log "所有任务均已提交给 auto_eval；调度器退出"
