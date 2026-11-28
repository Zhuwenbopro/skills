"""Find reproducible tuning candidates in a dense operator latency sweep."""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

from plot_sweep import read_sweep


def candidates(points, start, end, margin, split_independent):
    # At each M retain the least expensive previously measured equal-size split.
    splits = {}
    if split_independent:
        for part in range(start, end // 2 + 1):
            for total in range(2 * part, end + 1, part):
                estimate = (total // part) * points[part].optimized_us
                if total not in splits or estimate < splits[total][0]:
                    splits[total] = estimate, part

    fastest_larger = None
    for m in range(end, start - 1, -1):
        point = points[m]
        latency, floor, mem = point.optimized_us, point.floor_us, point.mem_us
        if m <= 64 and mem is not None:
            efficiency = floor / mem
            if efficiency < 0.90:
                yield m, "memory_floor_efficiency_lt_90pct", "", floor, mem, efficiency
            elif efficiency > 1.0:
                yield m, "memory_floor_efficiency_gt_100pct", "", floor, mem, efficiency
        if fastest_larger is not None:
            ref_us, ref_m = fastest_larger
            if latency > ref_us * (1 + margin):
                yield m, "smaller_slower", ref_m, ref_us, latency, latency / ref_us
        if fastest_larger is None or latency <= fastest_larger[0]:
            fastest_larger = latency, m
        if start < m < end:
            neighbor = max(points[m - 1].optimized_us, points[m + 1].optimized_us)
            if latency > neighbor * (1 + margin):
                yield m, "isolated_spike", f"{m-1},{m+1}", neighbor, latency, latency / neighbor
        if m in splits:
            ref_us, part = splits[m]
            if latency > ref_us * (1 + margin):
                yield m, "slower_than_equal_splits", part, ref_us, latency, latency / ref_us


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--noise-pct", type=float, default=0.0,
                        help="Relative margin above comparison latency, in percent")
    parser.add_argument("--split-independent", action="store_true",
                        help="Check integer multiples when splitting M yields equivalent independent outputs")
    args = parser.parse_args()
    if args.start < 1 or args.end < args.start or not math.isfinite(args.noise_pct) or args.noise_pct < 0:
        parser.error("invalid M range or noise margin")
    if args.out.resolve() == args.csv.resolve():
        parser.error("input and output must use different paths")
    try:
        groups = read_sweep(args.csv, args.start, args.end)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    counts = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("op", "m", "signal", "ref_m", "reference_us", "observed_us", "ratio"))
        for op, series in groups.items():
            points = {point.m: point for point in series}
            for row in candidates(points, args.start, args.end, args.noise_pct / 100, args.split_independent):
                writer.writerow((op, *row))
                counts[row[1]] += 1
    print(f"{sum(counts.values())} candidates -> {args.out}  {dict(counts)}")


if __name__ == "__main__":
    main()
