#!/usr/bin/env python3
"""Summarize comparison task lifecycle and EvalScope report metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

FAILURES = (
    "SGLang Server 启动失败",
    "SGLang Server 启动超过",
    "评测期间 SGLang Server 意外退出",
    "EvalScope 评测失败",
    "错误：",
)


def report_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/reports/*/*.json"))


def worker_is_alive(path: Path) -> bool:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        return not stat.is_file() or stat.read_text(encoding="utf-8").split()[2] != "Z"
    except (OSError, ValueError, IndexError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    run_root = args.plan.parent
    rows = []
    all_terminal = True

    for variant in plan["variants"]:
        label = variant["label"]
        task_dir = run_root / "tasks" / label
        log_path = task_dir / "auto_eval.log"
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        reports = report_files(run_root / "results" / label)
        metrics = []
        for path in reports:
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                metrics.append({
                    "dataset": report.get("dataset_name", path.stem),
                    "score": report.get("score"),
                    "metrics": {
                        metric.get("name", "unknown"): metric.get("score")
                        for metric in report.get("metrics", [])
                    },
                    "perf": report.get("perf_metrics", {}).get("summary", {}),
                    "report": str(path),
                })
            except (OSError, json.JSONDecodeError):
                continue

        failure = next((line for line in log.splitlines() if any(marker in line for marker in FAILURES)), "")
        if metrics and "EvalScope 评测成功" in log:
            state = "success"
        elif failure:
            state = "failed"
        elif (task_dir / "launcher.exit").is_file():
            exit_code = (task_dir / "launcher.exit").read_text(encoding="utf-8").strip()
            state = "failed" if exit_code != "0" else "running"
            if exit_code != "0":
                failure = f"auto_eval launcher 退出码={exit_code}"
            elif not worker_is_alive(task_dir / "worker.pid"):
                state = "failed"
                failure = "评测 worker 已退出，但未生成成功报告或标准失败标记"
            else:
                all_terminal = False
        elif "EvalScope 已启动" in log:
            state = "running"
            all_terminal = False
        elif "SGLang Server 已启动" in log:
            state = "starting_server"
            all_terminal = False
        elif "自动评测等待器启动" in log:
            state = "waiting_for_gpus"
            all_terminal = False
        else:
            state = "pending"
            all_terminal = False
        rows.append({
            "label": label,
            "parameters": variant["parameters"],
            "state": state,
            "failure": failure,
            "metrics": metrics,
            "log": str(log_path),
        })

    summary = {"all_terminal": all_terminal, "variant_count": len(rows), "variants": rows}
    output = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
