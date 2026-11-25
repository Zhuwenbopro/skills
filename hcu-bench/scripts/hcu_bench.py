"""Official Torch operator baselines for explicitly selected HCU devices."""
import argparse
import itertools
import json
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch


def summarize(values):
    if not values:
        return None
    return dict(n=len(values), median=statistics.median(values),
                mean=statistics.mean(values), min=min(values), max=max(values),
                stdev=statistics.stdev(values) if len(values) > 1 else 0.0)


def bandwidth_sample(byte_count, ms):
    return dict(ms=ms, GBps=byte_count / (ms / 1000) / 1e9,
                GiBps=byte_count / (ms / 1000) / 1024**3)


def synchronize(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def event_ms(fn, device, warmups, iters):
    stream = torch.cuda.Stream(device=device)
    # Tensor creation may have used the default stream.
    torch.cuda.synchronize(device)
    with torch.cuda.device(device), torch.cuda.stream(stream):
        for _ in range(warmups):
            fn()
        stream.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iters):
            fn()
        end.record(stream)
        end.synchronize()
        return start.elapsed_time(end) / iters


def memory_bandwidth(device, byte_count, args):
    src = torch.full((byte_count // 2,), 1, dtype=torch.bfloat16, device=f"cuda:{device}")
    dst = torch.zeros_like(src)
    samples = [bandwidth_sample(2 * byte_count, event_ms(
        lambda: dst.copy_(src), device, args.warmups, args.iters))
        for _ in range(args.repeats)]
    if not torch.equal(dst, src):
        raise RuntimeError("Same-device copy correctness check failed")
    return dict(device=device, status="ok", buffer_bytes=byte_count,
                byte_accounting="read_plus_write", timing="device_events",
                validation="passed", samples=samples,
                summary=summarize([s["GBps"] for s in samples]))


def p2p(dev0, dev1, byte_count, args):
    access = [bool(torch.cuda.can_device_access_peer(dev0, dev1)),
              bool(torch.cuda.can_device_access_peer(dev1, dev0))]
    result = dict(devices=[dev0, dev1], peer_access_both_directions=access,
                  buffer_bytes_per_direction=byte_count,
                  timing="host_wall_with_both_devices_synchronized")
    if not all(access):
        return dict(result, status="unsupported", reason="Direct peer access unavailable; no host-staging baseline substituted")
    src0 = torch.full((byte_count // 2,), 1, dtype=torch.bfloat16, device=f"cuda:{dev0}")
    src1 = torch.full_like(src0, 2, device=f"cuda:{dev1}")
    dst0, dst1 = torch.zeros_like(src0), torch.zeros_like(src1)
    stream0, stream1 = torch.cuda.Stream(device=dev0), torch.cuda.Stream(device=dev1)
    synchronize([dev0, dev1])

    def copy(dst, src, stream):
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)

    def forward():
        copy(dst1, src0, stream1)

    def reverse():
        copy(dst0, src1, stream0)

    def both():
        forward()
        reverse()

    def measure(fn, moved):
        samples = []
        for _ in range(args.repeats):
            for _ in range(args.warmups):
                fn()
            synchronize([dev0, dev1])
            begin = time.perf_counter()
            for _ in range(args.iters):
                fn()
            synchronize([dev0, dev1])
            ms = (time.perf_counter() - begin) * 1000 / args.iters
            samples.append(bandwidth_sample(moved, ms))
        return dict(samples=samples, summary=summarize([s["GBps"] for s in samples]))

    result["directions"] = {
        f"{dev0}_to_{dev1}": measure(forward, byte_count),
        f"{dev1}_to_{dev0}": measure(reverse, byte_count),
        "bidirectional_aggregate": measure(both, 2 * byte_count),
    }
    if not bool((dst0 == 2).all().item()) or not bool((dst1 == 1).all().item()):
        raise RuntimeError("P2P copy correctness check failed")
    return dict(result, status="ok", validation="passed")


def gemm(device, dtype, n, args):
    if dtype == "bf16":
        a = torch.randn((n, n), dtype=torch.bfloat16, device=f"cuda:{device}")
        b = torch.randn_like(a)
        op, impl, unit = lambda: torch.mm(a, b), "torch.mm", "TFLOPS"
    else:
        a = torch.randint(-127, 127, (n, n), dtype=torch.int8, device=f"cuda:{device}")
        b = torch.randint_like(a, -127, 127)
        int_mm = getattr(torch, "_int_mm", None) or torch.ops.aten._int_mm
        op, impl, unit = lambda: int_mm(a, b), "aten._int_mm", "TOPS"
    out = op()
    synchronize([device])
    errors = []
    for i, j in ((0, 0), (n // 2, n // 3), (n - 1, n - 1)):
        ref = (a[i, :].cpu().double() * b[:, j].cpu().double()).sum().item()
        actual = out[i, j].item()
        error = abs(actual - ref)
        errors.append(error)
        if (dtype == "int8" and error != 0) or (dtype == "bf16" and error > 1 + abs(ref) * .03):
            raise RuntimeError(f"{dtype} GEMM correctness failed at {i},{j}: {actual} vs {ref}")
    del out
    samples = []
    for _ in range(args.repeats):
        ms = event_ms(op, device, args.warmups, args.iters)
        samples.append(dict(ms=ms, throughput=2 * n**3 / (ms / 1000) / 1e12))
    return dict(device=device, dtype=dtype, n=n, impl=impl, unit=unit,
                layout="contiguous_row_major_A_and_B", status="ok",
                timing="device_events", validation=dict(sampled_entries=3,
                max_abs_error=max(errors), reference="CPU float64 dot products"),
                samples=samples, summary=summarize([s["throughput"] for s in samples]))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    ap.add_argument("--devices", required=True, help="Explicit Torch logical indices; one or more")
    ap.add_argument("--physical-devices", help="Matching host hy-smi indices, verified by PCI bus/UUID")
    ap.add_argument("--gemm-sizes", default="10240")
    ap.add_argument("--gemm-dtypes", default="bf16,int8")
    ap.add_argument("--memory-mib", type=int, default=512)
    ap.add_argument("--p2p-mib", type=int, default=256)
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--scope", default="unspecified")
    ap.add_argument("--hardware-label", help="Operator-supplied label; not a replacement for detected device name")
    ap.add_argument("--environment-json", help="Optional external preflight metadata JSON")
    args = ap.parse_args()
    try:
        args.device_list = [int(d) for d in args.devices.split(",")]
        args.size_list = [int(n) for n in args.gemm_sizes.split(",")]
        args.dtype_list = args.gemm_dtypes.split(",")
        if min(args.device_list) < 0 or len(set(args.device_list)) != len(args.device_list):
            raise ValueError("Device indices must be unique and nonnegative")
        if min(args.size_list) <= 0 or any(d not in ("bf16", "int8") for d in args.dtype_list):
            raise ValueError("Positive GEMM sizes and bf16/int8 dtypes required")
        if min(args.memory_mib, args.p2p_mib, args.iters, args.repeats) <= 0 or args.warmups < 0:
            raise ValueError("Positive buffers/iters/repeats and nonnegative warmups required")
        if args.physical_devices and len(args.physical_devices.split(",")) != len(args.device_list):
            raise ValueError("--physical-devices must match --devices length")
    except ValueError as exc:
        ap.error(str(exc))
    return args


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host=platform.node(),
                    python=sys.version, executable=sys.executable, torch=torch.__version__,
                    hip=torch.version.hip, scope=args.scope, parameters=vars(args),
                    environment={k: os.environ.get(k) for k in (
                        "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                        "HCU_ADAPTER_HOST", "SGLANG_PLUGINS", "ROCBLAS_TENSILE_LIBPATH")}, devices=[])
    result = dict(schema_version=2, status="running", metadata=metadata,
                  memory_bandwidth=[], p2p=[], gemm=[], errors=[])

    def save():
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        tmp.replace(output)

    def run_case(kind, details, fn):
        print(json.dumps(dict(stage=kind, **details)), flush=True)
        try:
            row = fn()
        except (NotImplementedError, RuntimeError) as exc:
            text = str(exc).lower()
            unsupported = any(t in text for t in ("not implemented", "not supported", "unsupported", "could not run 'aten::_int_mm'"))
            if kind == "gemm" and details.get("dtype") == "int8" and unsupported:
                row = dict(details, status="unsupported", reason=str(exc))
            else:
                raise
        result[kind].append(row)
        torch.cuda.empty_cache()
        save()

    save()
    try:
        if args.environment_json:
            metadata["external_environment"] = json.loads(Path(args.environment_json).read_text())
        torch.cuda.init()
        metadata["device_count"] = torch.cuda.device_count()
        mapping = args.physical_devices.split(",") if args.physical_devices else [None] * len(args.device_list)
        for d, physical in zip(args.device_list, mapping):
            with torch.cuda.device(d):
                free, total = torch.cuda.mem_get_info()
                props = torch.cuda.get_device_properties(d)
                row = dict(index=d, physical_index=physical, name=props.name, total_bytes=total,
                           free_bytes_before=free, properties_repr=str(props))
                for field in ("gcnArchName", "uuid", "pci_bus_id", "pci_device_id", "pci_domain_id", "multi_processor_count"):
                    row[field] = str(getattr(props, field, "unavailable"))
                metadata["devices"].append(row)
                needed = max(2 * args.memory_mib * 1024**2, 2 * args.p2p_mib * 1024**2,
                             6 * max(args.size_list)**2) + 512 * 1024**2
                if free < needed:
                    raise RuntimeError(f"Device {d}: free {free} below conservative requirement {needed}; reduce sizes explicitly")
        save()
        for d in args.device_list:
            run_case("memory_bandwidth", dict(device=d), lambda d=d: memory_bandwidth(d, args.memory_mib * 1024**2, args))
        for d0, d1 in itertools.combinations(args.device_list, 2):
            run_case("p2p", dict(devices=[d0, d1]), lambda d0=d0, d1=d1: p2p(d0, d1, args.p2p_mib * 1024**2, args))
        for d in args.device_list:
            for n in args.size_list:
                for dtype in args.dtype_list:
                    run_case("gemm", dict(device=d, n=n, dtype=dtype), lambda d=d, dtype=dtype, n=n: gemm(d, dtype, n, args))
        result["status"] = "completed_with_unsupported" if any(row["status"] == "unsupported" for kind in ("p2p", "gemm") for row in result[kind]) else "ok"
    except Exception as exc:
        result["status"] = "failed"
        result["errors"].append(dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    finally:
        for row in metadata["devices"]:
            try:
                with torch.cuda.device(row["index"]):
                    torch.cuda.empty_cache()
                    row["free_bytes_after"], _ = torch.cuda.mem_get_info()
            except Exception as exc:
                row["post_memory_error"] = str(exc)
        metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
    print(json.dumps(dict(status=result["status"], output=str(output))), flush=True)
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
