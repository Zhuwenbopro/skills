#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${AUTO_EVAL_CONFIG:-"${SCRIPT_DIR}/config.env"}

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "错误：配置文件不存在：$CONFIG_FILE" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

log() {
  printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*"
}

die() {
  log "错误：$*" >&2
  exit 2
}

resolve_from_script_dir() {
  local path=$1
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$SCRIPT_DIR" "$path"
  fi
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

: "${HOST:=127.0.0.1}"
: "${HEALTH_HOST:=127.0.0.1}"
: "${SERVER_COMMAND:=server_command.sh}"
: "${EVAL_COMMAND:=eval_command.sh}"
: "${RESULT_ROOT:=eval_results}"
: "${START_PORT:=30000}"
: "${END_PORT:=30100}"
: "${SERVER_START_TIMEOUT:=600}"
: "${HEALTH_CHECK_INTERVAL:=2}"
: "${GPU_VRAM_MAX_PERCENT:=5}"
: "${GPU_HCU_MAX_PERCENT:=0}"
: "${GPU_POLL_INTERVAL:=30}"
: "${GPU_CONFIRM_SECONDS:=3}"
: "${GPU_ALLOWLIST:=}"
: "${RETRY_INTERVAL:=30}"
: "${MAX_ATTEMPTS:=0}"
: "${SHUTDOWN_TIMEOUT:=15}"
: "${FAILURE_LOG_LINES:=100}"
: "${GPU_LOCK_DIR:=/tmp/auto_eval_gpu_locks}"

SERVER_COMMAND=$(resolve_from_script_dir "$SERVER_COMMAND")
EVAL_COMMAND=$(resolve_from_script_dir "$EVAL_COMMAND")
RESULT_ROOT=$(resolve_from_script_dir "$RESULT_ROOT")
SERVER_PARSER="${SCRIPT_DIR}/lib/server_command_parser.py"
GPU_ALLOWLIST=${GPU_ALLOWLIST//[[:space:]]/}
[[ -z "$GPU_ALLOWLIST" || "$GPU_ALLOWLIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
  die "GPU_ALLOWLIST 必须是逗号分隔的 GPU 编号，例如 2,3,4,5"

[[ -f "$SERVER_COMMAND" ]] || die "服务命令文件不存在：$SERVER_COMMAND"
[[ -f "$EVAL_COMMAND" ]] || die "评测命令文件不存在：$EVAL_COMMAND"
[[ -f "$SERVER_PARSER" ]] || die "服务命令解析器不存在：$SERVER_PARSER"

for cmd in python3 curl rocm-smi setsid flock tail grep; do
  command -v "$cmd" >/dev/null 2>&1 || die "找不到命令：$cmd"
done

if ! SERVER_METADATA=$(python3 "$SERVER_PARSER" metadata "$SERVER_COMMAND"); then
  die "server_command.sh 解析失败"
fi
mapfile -t SERVER_META <<<"$SERVER_METADATA"
((${#SERVER_META[@]} == 4)) || die "服务命令解析器返回了异常结果"

MODEL_PATH=${SERVER_META[0]%/}
MODEL_NAME=${SERVER_META[1]}
TP_SIZE=${SERVER_META[2]}
PP_SIZE=${SERVER_META[3]}
REQUIRED_GPUS=$((TP_SIZE * PP_SIZE))

for value_name in TP_SIZE PP_SIZE REQUIRED_GPUS START_PORT END_PORT \
  SERVER_START_TIMEOUT HEALTH_CHECK_INTERVAL GPU_POLL_INTERVAL \
  SHUTDOWN_TIMEOUT FAILURE_LOG_LINES; do
  is_positive_integer "${!value_name}" || die "$value_name 必须是正整数"
done

for value_name in GPU_VRAM_MAX_PERCENT GPU_HCU_MAX_PERCENT \
  GPU_CONFIRM_SECONDS RETRY_INTERVAL MAX_ATTEMPTS; do
  is_nonnegative_integer "${!value_name}" || die "$value_name 必须是非负整数"
done

((START_PORT <= END_PORT)) || die "START_PORT 不能大于 END_PORT"

[[ -d "$MODEL_PATH" ]] || die "模型路径不存在：$MODEL_PATH"

mkdir -p "$RESULT_ROOT" "$GPU_LOCK_DIR"

SELF_PGID=$(python3 -c 'import os; print(os.getpgrp())')
ATTEMPT_NO=0
SERVER_PID=""
SERVER_PGID=""
EVAL_PID=""
EVAL_PGID=""
SERVER_LOG=""
EVAL_LOG=""
SELECTED_GPUS=()
GPU_LOCK_FDS=()
LAST_FREE_COUNT=0

find_free_port() {
  python3 - "$HOST" "$START_PORT" "$END_PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
start_port = int(sys.argv[2])
end_port = int(sys.argv[3])

for port in range(start_port, end_port + 1):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except OSError:
        continue
    else:
        print(port)
        raise SystemExit(0)
    finally:
        sock.close()

print(f"错误：{start_port}-{end_port} 内没有可用端口", file=sys.stderr)
raise SystemExit(1)
PY
}

query_free_gpus() {
  local smi_output
  if ! smi_output=$(rocm-smi 2>&1); then
    log "rocm-smi 执行失败：${smi_output}" >&2
    return 1
  fi

  printf '%s\n' "$smi_output" | awk \
    -v max_vram="$GPU_VRAM_MAX_PERCENT" \
    -v max_hcu="$GPU_HCU_MAX_PERCENT" '
      $1 ~ /^[0-9]+$/ {
        gpu=$1
        vram=$6
        hcu=$7
        gsub(/%/, "", vram)
        gsub(/%/, "", hcu)
        if ((vram + 0) < max_vram && (hcu + 0) <= max_hcu) {
          print gpu
        }
      }
    '
}

gpu_is_allowed() {
  local gpu=$1
  [[ -z "$GPU_ALLOWLIST" || ",$GPU_ALLOWLIST," == *",$gpu,"* ]]
}

release_gpu_locks() {
  local fd
  for fd in "${GPU_LOCK_FDS[@]:-}"; do
    [[ -n "$fd" ]] || continue
    flock -u "$fd" 2>/dev/null || true
    exec {fd}>&- 2>/dev/null || true
  done
  GPU_LOCK_FDS=()
  SELECTED_GPUS=()
}

select_and_lock_gpus() {
  local output gpu lock_fd
  local -a free_gpus=()

  release_gpu_locks
  if ! output=$(query_free_gpus); then
    LAST_FREE_COUNT=0
    return 1
  fi

  if [[ -n "$output" ]]; then
    mapfile -t free_gpus <<<"$output"
  fi
  LAST_FREE_COUNT=${#free_gpus[@]}

  for gpu in "${free_gpus[@]}"; do
    gpu_is_allowed "$gpu" || continue

    exec {lock_fd}>"${GPU_LOCK_DIR}/gpu_${gpu}.lock"
    if flock -n "$lock_fd"; then
      SELECTED_GPUS+=("$gpu")
      GPU_LOCK_FDS+=("$lock_fd")
    else
      exec {lock_fd}>&-
    fi

    if ((${#SELECTED_GPUS[@]} == REQUIRED_GPUS)); then
      return 0
    fi
  done

  release_gpu_locks
  return 1
}

selected_gpus_still_free() {
  local output gpu
  if ! output=$(query_free_gpus); then
    return 1
  fi

  for gpu in "${SELECTED_GPUS[@]}"; do
    if ! grep -qx -- "$gpu" <<<"$output"; then
      return 1
    fi
  done
}

get_process_group() {
  local pid=$1 pgid
  pgid=$(python3 - "$pid" 2>/dev/null <<'PY' || true
import os
import sys

print(os.getpgid(int(sys.argv[1])))
PY
  )
  printf '%s\n' "${pgid:-$pid}"
}

terminate_process_group() {
  local name=$1 pid=$2 pgid=$3
  local deadline target

  [[ -n "$pid" ]] || return 0

  if [[ "$pgid" =~ ^[0-9]+$ && "$pgid" != "$SELF_PGID" ]]; then
    target="-$pgid"
  else
    target="$pid"
  fi

  if kill -0 -- "$target" 2>/dev/null; then
    log "停止${name}，PID=${pid}，PGID=${pgid}"
    kill -TERM -- "$target" 2>/dev/null || true
    deadline=$((SECONDS + SHUTDOWN_TIMEOUT))
    while kill -0 -- "$target" 2>/dev/null && ((SECONDS < deadline)); do
      sleep 1
    done
    if kill -0 -- "$target" 2>/dev/null; then
      log "${name}未在 ${SHUTDOWN_TIMEOUT}s 内退出，发送 KILL"
      kill -KILL -- "$target" 2>/dev/null || true
    fi
  fi

  wait "$pid" 2>/dev/null || true
}

cleanup_current_attempt() {
  terminate_process_group " EvalScope" "$EVAL_PID" "$EVAL_PGID"
  EVAL_PID=""
  EVAL_PGID=""
  terminate_process_group " SGLang Server" "$SERVER_PID" "$SERVER_PGID"
  SERVER_PID=""
  SERVER_PGID=""
  release_gpu_locks
}

on_signal() {
  local signal=$1 code=$2
  log "收到 ${signal}，正在释放本轮资源"
  exit "$code"
}

trap cleanup_current_attempt EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

start_server() {
  setsid python3 "$SERVER_PARSER" run "$SERVER_COMMAND" \
    >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  SERVER_PGID=$(get_process_group "$SERVER_PID")
  log "SGLang Server 已启动：PID=${SERVER_PID}，PGID=${SERVER_PGID}"
}

wait_for_server() {
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))

  while ((SECONDS < deadline)); do
    if curl -fsS "http://${HEALTH_HOST}:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi

    sleep "$HEALTH_CHECK_INTERVAL"
  done

  return 124
}

start_eval() {
  setsid bash "$EVAL_COMMAND" >"$EVAL_LOG" 2>&1 &
  EVAL_PID=$!
  EVAL_PGID=$(get_process_group "$EVAL_PID")
  log "EvalScope 已启动：PID=${EVAL_PID}，PGID=${EVAL_PGID}"
}

wait_for_eval() {
  local status

  while kill -0 "$EVAL_PID" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "评测期间 SGLang Server 意外退出"
      return 125
    fi
    sleep 2
  done

  if wait "$EVAL_PID"; then
    status=0
  else
    status=$?
  fi
  EVAL_PID=""
  EVAL_PGID=""
  return "$status"
}

show_log_tail() {
  local title=$1 path=$2
  [[ -f "$path" ]] || return 0
  log "${title}最近 ${FAILURE_LOG_LINES} 行："
  tail -n "$FAILURE_LOG_LINES" "$path" || true
}

run_one_attempt() {
  local selected_csv time_tag status

  if ! PORT=$(find_free_port); then
    log "本轮没有找到可用端口"
    return 1
  fi

  selected_csv=$(IFS=,; printf '%s' "${SELECTED_GPUS[*]}")
  time_tag=$(date +'%Y%m%d_%H%M%S')
  RUN_DIR="${RESULT_ROOT}/${MODEL_NAME}/${time_tag}_attempt${ATTEMPT_NO}"
  SERVER_LOG="${RUN_DIR}/server.log"
  EVAL_LOG="${RUN_DIR}/eval.log"
  mkdir -p "$RUN_DIR"

  export MODEL_PATH MODEL_NAME TP_SIZE PP_SIZE HOST HEALTH_HOST PORT RUN_DIR
  export HIP_VISIBLE_DEVICES="$selected_csv"

  log "第 ${ATTEMPT_NO} 次尝试：GPU=${selected_csv}，PORT=${PORT}"
  log "本轮目录：$RUN_DIR"
  start_server

  if wait_for_server; then
    log "SGLang 服务健康检查通过：http://${HEALTH_HOST}:${PORT}"
  else
    status=$?
    if ((status == 124)); then
      log "SGLang Server 启动超过 ${SERVER_START_TIMEOUT}s"
    else
      log "SGLang Server 启动失败，退出状态=${status}"
    fi
    show_log_tail "服务日志" "$SERVER_LOG"
    return 10
  fi

  start_eval
  if wait_for_eval; then
    log "EvalScope 评测成功"
    return 0
  else
    status=$?
    log "EvalScope 评测失败，退出状态=${status}"
    show_log_tail "评测日志" "$EVAL_LOG"
    show_log_tail "服务日志" "$SERVER_LOG"
    return 20
  fi
}

log "自动评测等待器启动"
log "模型：${MODEL_NAME}（${MODEL_PATH}）"
log "需要 GPU：${REQUIRED_GPUS} 张（TP=${TP_SIZE}, PP=${PP_SIZE}）"
[[ -n "$GPU_ALLOWLIST" ]] && log "GPU 候选范围：${GPU_ALLOWLIST}"

while true; do
  if ! select_and_lock_gpus; then
    log "当前符合空闲标准的 GPU：${LAST_FREE_COUNT} 张；需要：${REQUIRED_GPUS} 张"
    sleep "$GPU_POLL_INTERVAL"
    continue
  fi

  if ((GPU_CONFIRM_SECONDS > 0)); then
    log "发现候选 GPU：${SELECTED_GPUS[*]}，${GPU_CONFIRM_SECONDS}s 后再次确认"
    sleep "$GPU_CONFIRM_SECONDS"
  fi

  if ! selected_gpus_still_free; then
    log "候选 GPU 状态发生变化，重新等待"
    release_gpu_locks
    sleep "$GPU_POLL_INTERVAL"
    continue
  fi

  ATTEMPT_NO=$((ATTEMPT_NO + 1))
  if run_one_attempt; then
    cleanup_current_attempt
    trap - EXIT
    log "全部评测完成，脚本正常退出"
    exit 0
  else
    status=$?
    cleanup_current_attempt
    log "第 ${ATTEMPT_NO} 次尝试失败（状态=${status}），资源已释放"
  fi

  if ((MAX_ATTEMPTS > 0 && ATTEMPT_NO >= MAX_ATTEMPTS)); then
    trap - EXIT
    die "已达到最大尝试次数：${MAX_ATTEMPTS}"
  fi

  sleep "$RETRY_INTERVAL"
done
