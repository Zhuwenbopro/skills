"""Plot dense, fixed-clock HCU operator sweeps from measured CSV rows."""

import argparse
import csv
import gzip
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Point:
    m: int
    floor_us: float
    optimized_us: float
    baseline_us: float | None
    ops: float | None
    peak_tops: float | None
    mem_us: float | None = None

    @property
    def utilization(self):
        # TOP/s times microseconds gives millions of operations.
        return self.ops / (self.optimized_us * self.peak_tops * 1e6)

    @property
    def speedup(self):
        return self.baseline_us / self.optimized_us


def number(row, name, optional=False):
    raw = (row.get(name) or "").strip()
    if optional and not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} needs a number, got {raw!r}") from exc
    if not math.isfinite(value) or value < 0 or (value == 0 and name != "ops"):
        raise ValueError(f"invalid value for {name}: {raw!r}")
    return value


def read_sweep(path, start, end):
    if not 1 <= start <= end:
        raise ValueError("expected 1 <= start <= end")
    groups = defaultdict(dict)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"op", "m", "floor_us", "optimized_us"}
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")
        if len(set(fields)) != len(fields):
            raise ValueError("duplicate CSV column names")
        for row in reader:
            try:
                if None in row:
                    raise ValueError("too many fields")
                op = (row["op"] or "").strip()
                if not op:
                    raise ValueError("empty op name")
                m = int(row["m"] or "")
                if not start <= m <= end:
                    raise ValueError(f"{op}: M={m} is outside [{start}, {end}]")
                if m in groups[op]:
                    raise ValueError(f"{op}: duplicate M={m}")
                p = Point(
                    m, number(row, "floor_us"), number(row, "optimized_us"),
                    number(row, "baseline_us", True), number(row, "ops", True),
                    number(row, "peak_tops", True), number(row, "mem_us", True),
                )
                if (p.ops is None) != (p.peak_tops is None):
                    raise ValueError("ops and peak_tops must be supplied together")
                groups[op][m] = p
            except ValueError as exc:
                raise ValueError(f"{path}, line {reader.line_num}: {exc}") from exc
    if not groups:
        raise ValueError("CSV contains no measurements")
    sweeps = {}
    for op, by_m in groups.items():
        if len(by_m) != end - start + 1:
            missing_m = next(m for m in range(start, end + 1) if m not in by_m)
            raise ValueError(f"{op}: missing M={missing_m}; expected every integer")
        points = [by_m[m] for m in range(start, end + 1)]
        if len({p.baseline_us is None for p in points}) != 1:
            raise ValueError(f"{op}: baseline_us must cover every M when compared")
        if len({p.peak_tops for p in points}) != 1:
            raise ValueError(f"{op}: provide one sustained peak_tops for all M or omit compute data")
        sweeps[op] = points
    return sweeps


def draw(sweeps, out_dir, title, baseline_label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    matplotlib.rcParams["path.simplify"] = False
    out_dir.mkdir(parents=True, exist_ok=True)
    compared = {op: points for op, points in sweeps.items() if points[0].baseline_us is not None}
    speedup_path = out_dir / "speedup.png"
    if not compared and speedup_path.exists():
        raise ValueError(f"no comparison in this sweep; use a fresh --out-dir to avoid stale {speedup_path}")

    def panels(count, heading):
        fig, axes = plt.subplots(count, 1, figsize=(12, 3.4 * count + 0.65),
                                 sharex=True, squeeze=False, layout="constrained")
        fig.suptitle(f"{title}  |  {heading}" if title else heading, fontsize=16)
        axes[-1, 0].set_xlabel("Batch / M")
        return fig, axes[:, 0]

    def style(ax, op, ms):
        ax.set_title(op, loc="left")
        ax.grid(axis="both", alpha=0.17)
        ax.spines["top"].set_visible(False)
        ax.set_xlim((ms[0], ms[-1]) if len(ms) > 1 else (ms[0] - 0.5, ms[0] + 0.5))

    fig, axes = panels(len(sweeps), "Latency")
    for ax, (op, points) in zip(axes, sweeps.items()):
        ms = [p.m for p in points]
        marker = "o" if len(ms) == 1 else None
        style(ax, op, ms)
        handles = []
        if points[0].ops is not None:
            util = [p.utilization for p in points]
            right = ax.twinx()
            right.set_ylim(0, max(1.0, max(util)))
            right.yaxis.set_major_formatter(PercentFormatter(1.0))
            right.set_ylabel("Compute utilization", color="#497389")
            right.tick_params(axis="y", colors="#497389")
            right.spines["top"].set_visible(False)
            handles.append(right.fill_between(ms, util, color="#c6dded", alpha=0.72,
                                              label=f"Compute / {points[0].peak_tops:g} TOP/s"))
            ax.set_zorder(right.get_zorder() + 1)
            ax.patch.set_visible(False)
        if op in compared:
            handles += ax.plot(ms, [p.baseline_us for p in points], color="#276b89",
                               marker=marker, label=baseline_label)
        handles += ax.plot(ms, [p.optimized_us for p in points], color="#bd4b1e",
                           marker=marker, label="Optimized")
        handles += ax.plot(ms, [p.floor_us for p in points], color="#76568d",
                           marker=marker, linestyle="--", label="Memory floor")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Latency (µs)")
        ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2)
    fig.savefig(out_dir / "latency.png", dpi=160)
    plt.close(fig)
    if not compared:
        print("No measured baseline; speedup is unavailable.")
        return
    fig, axes = panels(len(compared), f"Speedup vs {baseline_label}")
    for ax, (op, points) in zip(axes, compared.items()):
        ms = [p.m for p in points]
        style(ax, op, ms)
        ax.axhline(1, linestyle="--", color="#666666", linewidth=0.9)
        ax.plot(ms, [p.speedup for p in points], color="#bd4b1e",
                marker="o" if len(ms) == 1 else None)
        ax.set_ylabel("Speedup (×)")
    fig.savefig(speedup_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="dense CSV or CSV.gz; one row per (op, m)")
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument("--baseline-label", default="Baseline")
    args = parser.parse_args()
    try:
        sweeps = read_sweep(args.csv, args.start, args.end)
        draw(sweeps, args.out_dir, args.title, args.baseline_label)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
