#!/usr/bin/env python3
"""Validate the generated HCU performance report bundle.

This is intentionally stdlib-only so it can run in the prepared server
environment. It checks the publication contract, not the truth of the trace.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def _norm(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        errors.append(f"missing file: {path.name}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        errors.append(f"invalid JSON {path.name}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"JSON root must be an object: {path.name}")
        return None
    return value


def _heading_block(text: str, pattern: str) -> str:
    match = re.search(rf"(?im)^#+\s*{pattern}.*$", text)
    if not match:
        return ""
    rest = text[match.end() :]
    next_heading = re.search(r"(?m)^#+\s+", rest)
    return rest[: next_heading.start()] if next_heading else rest


def _table_rows(block: str) -> tuple[list[str], list[list[str]], int]:
    lines = [line.strip() for line in block.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return [], [], 0
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[list[str]] = []
    malformed = 0
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(cells)
        else:
            malformed += 1
    return header, rows, malformed


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _close(left: float, right: float, rel: float = 2e-3, abs_tol: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=rel, abs_tol=abs_tol)


def validate(root: Path, write_validation: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    readme_path = root / "README.md"
    data_path = root / "analysis-data.json"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    if not readme:
        errors.append("missing or empty README.md")

    required_files = ["README.md", "analysis-data.json", "source-map.json", "coverage.json"]
    for name in required_files:
        if not (root / name).exists():
            errors.append(f"missing file: {name}")
    if not (root / "draft-verify-kernel-time.png").exists():
        errors.append("missing plot: draft-verify-kernel-time.png")

    data = _load_json(data_path, errors)
    active_data_stages: set[str] = set()
    if data is not None:
        for stage in ("draft", "verify"):
            if isinstance(data.get(stage), list) and data[stage]:
                active_data_stages.add(stage.upper())

    section_patterns = {
        "environment": r"运行环境与有效配置",
        "hardware": r"硬件实测参考",
        "method": r"测试方法",
    }
    for name, pattern in section_patterns.items():
        if not re.search(rf"(?im)^#+\s*{pattern}", readme):
            errors.append(f"missing required section: {name}")

    stage_tables: dict[str, dict[str, Any]] = {}
    required_headers = [
        "模块/函数",
        "归属类别",
        "输入/权重/输出shape",
        "次数/replay",
        "总耗时",
        "单次耗时",
        "算力利用率",
        "带宽利用率",
    ]
    for stage in ("DRAFT", "VERIFY"):
        has_heading = re.search(rf"(?im)^#+\s*{stage}\s+(?:算子)?调用顺序", readme)
        if not has_heading:
            if stage in active_data_stages:
                errors.append(f"missing {stage} call-order section")
            continue
        block = _heading_block(readme, rf"{stage}\s+(?:算子)?调用顺序")
        header, rows, malformed = _table_rows(block)
        if malformed:
            errors.append(f"{stage} table has {malformed} malformed data rows")
        normalized = {_norm(item) for item in header}
        missing = [
            item
            for item in required_headers
            if not any(
                _norm(item) == candidate
                or (item in ("总耗时", "单次耗时", "次数/replay") and candidate.startswith(_norm(item)))
                for candidate in normalized
            )
        ]
        if missing:
            errors.append(f"{stage} table missing columns: {', '.join(missing)}")
        count_headers = [candidate for candidate in normalized if candidate.startswith(_norm("次数/replay"))]
        if count_headers and not any("单step" in candidate for candidate in count_headers):
            errors.append(f"{stage} table count column must be normalized to one step")
        if len(rows) < 8:
            errors.append(f"{stage} table has {len(rows)} detail rows; at least 8 are required")
        if len(rows) < 15:
            warnings.append(f"{stage} table has only {len(rows)} rows; reference reports usually have dozens")
        module_index = next((i for i, item in enumerate(header) if _norm(item) == _norm("模块/函数")), None)
        if module_index is not None:
            modules = [row[module_index] for row in rows]
            annotation_only = all(
                re.match(
                    r"(?i)^(?:draft(?:\s|$)|draft_extend(?:\s|$)|draft_extend_v2(?:\s|$)|target_verify(?:\s|$)|step\[)",
                    module.strip(),
                )
                for module in modules
            )
            if annotation_only:
                errors.append(f"{stage} table is annotation-only; model call chain is not expanded")
        stage_tables[stage] = {"headers": header, "rows": rows}

    auxiliary_errors: list[str] = []
    source_map = _load_json(root / "source-map.json", auxiliary_errors) if (root / "source-map.json").exists() else None
    coverage = _load_json(root / "coverage.json", auxiliary_errors) if (root / "coverage.json").exists() else None
    errors.extend(auxiliary_errors)
    if source_map is not None and not source_map:
        errors.append("source-map.json is empty; module/kernel/shape evidence is required")
    if coverage is not None and not coverage:
        errors.append("coverage.json is empty; mapped/unmapped/unknown coverage must be recorded")
    if coverage:
        coverage_status = str(coverage.get("status", coverage.get("coverage", ""))).lower()
        if any(token in coverage_status for token in ("not_proven", "unknown", "incomplete", "unverified")):
            errors.append(f"coverage status is not proven: {coverage_status}")
    if data is not None:
        for key in ("draft", "verify", "pie_data_us", "step_timing", "primary_metric"):
            if key not in data:
                errors.append(f"analysis-data.json missing key: {key}")
        scope = data.get("scope")
        if not isinstance(scope, dict) or scope.get("annotation_category") != "gpu_user_annotation":
            errors.append("analysis-data.json must record annotation_category=gpu_user_annotation for Graph stage timing")
        if not isinstance(scope, dict) or not scope.get("nested_scope_dedup"):
            errors.append("analysis-data.json must record nested GPU scope de-duplication")
        if data.get("primary_metric") and "GPU kernel duration" not in str(data["primary_metric"]):
            warnings.append("primary_metric does not state GPU kernel duration")

        stage_sums: dict[str, float] = {}
        stage_category_sums: dict[str, dict[str, float]] = {}
        for stage in ("draft", "verify"):
            rows = data.get(stage)
            if not isinstance(rows, list) or not rows:
                errors.append(f"analysis-data.json {stage} must be a non-empty array")
                continue
            total = 0.0
            category_sums: dict[str, float] = {}
            table = stage_tables.get(stage.upper(), {})
            category_column = next((i for i, h in enumerate(table.get("headers", [])) if _norm(h) == _norm("归属类别")), None)
            table_rows = table.get("rows", [])
            if len(table_rows) != len(rows):
                errors.append(f"{stage} table rows do not match publication data")
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    errors.append(f"analysis-data.json {stage}[{index}] must be an object")
                    continue
                for key in ("time_ms", "calls", "single_call_ms", "categories"):
                    if key not in row:
                        errors.append(f"analysis-data.json {stage}[{index}] missing {key}")
                time_ms = _number(row.get("time_ms"))
                calls = _number(row.get("calls"))
                single = _number(row.get("single_call_ms"))
                if time_ms is not None and calls and single is not None and not _close(time_ms / calls, single):
                    errors.append(f"{stage}[{index}] single_call_ms does not match time_ms/calls")
                categories = row.get("categories")
                category = row.get("category")
                if category not in ("GEMM", "Element-wise", "Other", "Communication"):
                    errors.append(f"{stage}[{index}] has missing or invalid category")
                if isinstance(categories, dict) and set(categories) != {category}:
                    errors.append(f"{stage}[{index}] category does not match category time data")
                if category_column is not None and index < len(table_rows) and table_rows[index][category_column] != category:
                    errors.append(f"{stage}[{index}] table category does not match publication data")
                if time_ms is not None:
                    total += time_ms
                    if isinstance(category, str):
                        category_sums[category] = category_sums.get(category, 0.0) + time_ms * 1000.0
                if isinstance(categories, dict) and time_ms is not None:
                    category_us = sum(_number(item.get("time_us")) or 0.0 for item in categories.values() if isinstance(item, dict))
                    if not _close(category_us / 1000.0, time_ms):
                        errors.append(f"{stage}[{index}] category time does not match row time")
            stage_sums[stage] = total
            stage_category_sums[stage] = category_sums

        pie = data.get("pie_data_us")
        if isinstance(pie, dict):
            for stage in ("DRAFT", "VERIFY"):
                values = pie.get(stage)
                if not isinstance(values, list) or not values:
                    errors.append(f"pie_data_us missing {stage}")
                    continue
                if any((_number(value) or 0.0) < 0 for value in values):
                    errors.append(f"pie_data_us {stage} contains negative values")
                if not any((_number(value) or 0.0) > 0 for value in values):
                    errors.append(f"pie_data_us {stage} is all zero")
                if stage.lower() in stage_sums and not _close(sum(_number(value) or 0.0 for value in values) / 1000.0, stage_sums[stage.lower()]):
                    errors.append(f"pie_data_us {stage} does not match {stage.lower()} row totals")
                labels = data.get("pie_categories", [])
                if len(labels) != len(values):
                    errors.append(f"pie category labels do not match {stage} values")
                else:
                    category_sums = stage_category_sums.get(stage.lower(), {})
                    for label, value in zip(labels, values):
                        if not _close(category_sums.get(label, 0.0), _number(value) or 0.0):
                            errors.append(f"pie_data_us {stage}/{label} does not match row category totals")

        timing = data.get("step_timing")
        if isinstance(timing, dict):
            for key in ("draft_kernel_per_cycle_ms", "verify_kernel_per_cycle_ms", "kernel_sum_per_cycle_ms"):
                if key not in timing:
                    errors.append(f"step_timing missing {key}")
            draft_cycle = _number(timing.get("draft_kernel_per_cycle_ms"))
            verify_cycle = _number(timing.get("verify_kernel_per_cycle_ms"))
            combined = _number(timing.get("kernel_sum_per_cycle_ms"))
            if draft_cycle is not None and verify_cycle is not None and combined is not None and not _close(draft_cycle + verify_cycle, combined):
                errors.append("kernel_sum_per_cycle_ms does not equal DRAFT + VERIFY")

    result = {
        "status": "complete" if not errors else "incomplete",
        "errors": errors,
        "warnings": warnings,
        "checked_files": sorted(path.name for path in root.iterdir() if path.is_file()) if root.exists() else [],
        "stage_rows": {stage: len(value["rows"]) for stage, value in stage_tables.items()},
    }
    if write_validation:
        (root / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--write-validation", action="store_true")
    args = parser.parse_args()
    result = validate(args.report_dir, args.write_validation)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
