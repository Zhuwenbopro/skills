#!/usr/bin/env python3
"""Hold memory on explicitly selected HCU devices until SIGTERM or SIGINT."""
import argparse
import json
import os
from pathlib import Path
import signal
import threading


def reservation_bytes(free, total, fraction, reserve_bytes):
    if not 0 < fraction < 1 or reserve_bytes < 0 or total <= 0 or not 0 <= free <= total:
        raise ValueError("invalid memory budget")
    target = min(int(total * fraction), free - reserve_bytes)
    if target < total * 0.80:
        raise RuntimeError("Insufficient free memory for an idle-card reservation")
    return target


def allocate(torch, index, target, chunk_bytes, stop, blocks):
    held = 0
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    with torch.cuda.device(index):
        while held < target and not stop.is_set():
            size = min(chunk_bytes, target - held)
            block = torch.empty(size, dtype=torch.uint8, device=f"cuda:{index}")
            blocks.append(block)
            block.fill_(1)
            held += size
        torch.cuda.synchronize(index)
    return held


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", required=True, help="Physical IDs, e.g. 0,1,2,3; inspect hy-smi first")
    p.add_argument("--fraction", type=float, default=0.90)
    p.add_argument("--reserve-gib", type=float, default=2.0)
    p.add_argument("--chunk-mib", type=int, default=256)
    p.add_argument("--pid-file", type=Path, default=Path("/home/hcu-vram-reservation.pid"))
    args = p.parse_args()
    try:
        devices = [int(x) for x in args.devices.split(",")]
    except ValueError:
        p.error("devices must be comma-separated integers")
    if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
        p.error("devices must be unique nonnegative IDs")
    if not 0.80 <= args.fraction < 1 or args.reserve_gib < 0 or args.chunk_mib <= 0:
        p.error("fraction must be in [0.80,1), reserve nonnegative and chunk positive")
    import fcntl
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    lock = open(str(args.pid_file) + ".lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("another holder owns this PID-file lock")
    if args.pid_file.exists():
        previous = int(args.pid_file.read_text().strip())
        if previous <= 0:
            p.error("invalid existing PID file")
        try:
            os.kill(previous, 0)
        except ProcessLookupError:
            pass
        else:
            p.error(f"PID file still refers to running process {previous}")
    os.environ["ROCR_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        os.environ.pop(key, None)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != len(devices):
        raise RuntimeError("visible device count does not match physical selection")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    blocks = []
    args.pid_file.write_text(str(os.getpid()) + "\n")
    print(json.dumps({"pid": os.getpid(), "devices": devices}), flush=True)
    try:
        for logical, physical in enumerate(devices):
            free, total = torch.cuda.mem_get_info(logical)
            target = reservation_bytes(free, total, args.fraction, int(args.reserve_gib * 2**30))
            held = allocate(torch, logical, target, args.chunk_mib * 2**20, stop, blocks)
            print(json.dumps({"device": physical, "held_gib": held / 2**30, "total_gib": total / 2**30}), flush=True)
            if stop.is_set():
                return
        print("READY: holding memory without a steady-state compute workload", flush=True)
        stop.wait()
    finally:
        blocks.clear()
        try:
            for i in range(len(devices)):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        finally:
            if args.pid_file.exists() and args.pid_file.read_text().strip() == str(os.getpid()):
                args.pid_file.unlink()
            lock.close()
        print("RELEASED", flush=True)


if __name__ == "__main__":
    main()
