#!/usr/bin/env python3
"""Generate safe SGLang command variants for a comparison matrix."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import shlex
import sys
from pathlib import Path

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
UNSET = "__UNSET__"


def die(message: str) -> "NoReturn":
    raise SystemExit(f"错误：{message}")


def parse_matrix(items: list[str]) -> list[tuple[str, list[str]]]:
    matrix: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for item in items:
        if "=" not in item:
            die(f"对照参数必须是 NAME=value1,value2：{item}")
        name, raw_values = item.split("=", 1)
        if not NAME_RE.fullmatch(name):
            die(f"非法环境变量名：{name}")
        if name in seen:
            die(f"对照参数重复：{name}")
        values = raw_values.split(",")
        if not values or any(value == "" for value in values):
            die(f"{name} 至少需要一个非空值")
        if len(set(values)) != len(values):
            die(f"{name} 包含重复值")
        seen.add(name)
        matrix.append((name, values))
    if not matrix:
        die("至少需要一个 --matrix 参数")
    return matrix


def parse_source(path: Path) -> tuple[list[str], list[tuple[str, str]], set[str], list[str]]:
    exports: dict[str, str] = {}
    unsets: set[str] = set()
    settings: list[str] = []
    assignments: list[tuple[str, str]] = []
    command: list[str] = []
    started = False
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not started and stripped in {"set -e", "set -u", "set -eu", "set -euo pipefail"}:
            settings.append(stripped)
            continue
        if not started and stripped.startswith("export "):
            try:
                tokens = shlex.split(stripped, comments=True, posix=True)
            except ValueError as exc:
                die(f"{path}:{line_no}: export 语法错误：{exc}")
            for token in tokens[1:]:
                if "=" not in token:
                    die(f"{path}:{line_no}: export 只支持 NAME=value")
                name, value = token.split("=", 1)
                if not NAME_RE.fullmatch(name):
                    die(f"{path}:{line_no}: 非法环境变量名：{name}")
                assignments = [(old_name, old_value) for old_name, old_value in assignments if old_name != name]
                assignments.append((name, value))
                exports[name] = value
                unsets.discard(name)
            continue
        if not started and stripped.startswith("unset "):
            try:
                tokens = shlex.split(stripped, comments=True, posix=True)
            except ValueError as exc:
                die(f"{path}:{line_no}: unset 语法错误：{exc}")
            if len(tokens) < 2 or any(not NAME_RE.fullmatch(name) for name in tokens[1:]):
                die(f"{path}:{line_no}: unset 只支持环境变量名")
            for name in tokens[1:]:
                unsets.add(name)
                exports.pop(name, None)
                assignments = [(old_name, old_value) for old_name, old_value in assignments if old_name != name]
            continue
        if not started and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", stripped):
            try:
                tokens = shlex.split(stripped, comments=True, posix=True)
            except ValueError as exc:
                die(f"{path}:{line_no}: 变量赋值语法错误：{exc}")
            if len(tokens) != 1 or "=" not in tokens[0]:
                die(f"{path}:{line_no}: 变量赋值必须单独占一行")
            name, value = tokens[0].split("=", 1)
            assignments = [(old_name, old_value) for old_name, old_value in assignments if old_name != name]
            assignments.append((name, value))
            exports[name] = value
            unsets.discard(name)
            continue
        started = True
        command.append(raw_line)
    if not command:
        die(f"{path}: 没有找到启动命令")
    return settings, assignments, unsets, command


def quote(value: str) -> str:
    return shlex.quote(value)


def safe_label(index: int, values: dict[str, str]) -> str:
    readable = "__".join(
        f"{name.lower()}-{('unset' if value == UNSET else value)}" for name, value in values.items()
    )
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", readable).strip("-_")
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:8]
    return f"{index:03d}_{readable[:100]}_{digest}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--matrix", action="append", default=[])
    args = parser.parse_args()

    if not args.source.is_file():
        die(f"命令模板不存在：{args.source}")
    matrix = parse_matrix(args.matrix)
    settings, base_assignments, base_unsets, command = parse_source(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    records = []
    value_lists = [values for _, values in matrix]
    for index, combination in enumerate(itertools.product(*value_lists), 1):
        selected = {matrix[pos][0]: value for pos, value in enumerate(combination)}
        assignments = list(base_assignments)
        unsets = set(base_unsets)
        for name, value in selected.items():
            assignments = [(old_name, old_value) for old_name, old_value in assignments if old_name != name]
            if value == UNSET:
                unsets.add(name)
            else:
                assignments.append((name, value))
                unsets.discard(name)

        label = safe_label(index, selected)
        command_path = (args.output_dir / f"{label}.sh").resolve()
        lines = ["#!/usr/bin/env bash", *settings]
        lines.extend(f"export {name}={quote(value)}" for name, value in assignments)
        lines.extend(f"unset {name}" for name in sorted(unsets))
        lines.extend(command)
        command_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        records.append({"id": index, "label": label, "parameters": selected, "server_command": str(command_path)})

    manifest = {
        "source": str(args.source.resolve()),
        "variables": [name for name, _ in matrix],
        "variant_count": len(records),
        "variants": records,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(len(records))


if __name__ == "__main__":
    main()
