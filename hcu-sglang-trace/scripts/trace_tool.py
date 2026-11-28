import argparse
import collections
import copy
import os
import gzip
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream)


def trace_paths(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.trace.json*") if p.is_file() and p.name.endswith((".trace.json", ".trace.json.gz")))


def family(name):
    if "nccl" in name.lower():
        return "RCCL"
    if name.startswith("Cijk"):
        return "GEMM_including_PostGSU"
    if "index_put_kernel_impl" in name:
        return "index_write"
    if "silu" in name.lower():
        return "SiLU"
    if "norm" in name.lower():
        return "normalization"
    return "other"


def inspect_trace(path, relative_root, args):
    data = read_json(path)
    events = data["traceEvents"] if isinstance(data, dict) else data
    kernels = [e for e in events if e.get("ph") == "X" and "kernel" in e.get("cat", "").lower()]
    launches = [e for e in events if e.get("ph") == "X" and "graphlaunch" in e.get("name", "").lower()]
    correlations = [e.get("args", {}).get("correlation") for e in launches]
    valid_ids = {cid for cid in correlations if cid is not None}
    groups = collections.defaultdict(list)
    for e in kernels:
        cid = e.get("args", {}).get("correlation")
        if cid in valid_ids:
            groups[cid].append(e)
    counts = [len(groups[cid]) if cid is not None else 0 for cid in correlations]
    graph_kernels = [e for cid in valid_ids for e in groups[cid]]
    streams = collections.Counter(str(e.get("args", {}).get("stream", e.get("tid"))) for e in graph_kernels)
    comm_streams = collections.Counter(str(e.get("args", {}).get("stream", e.get("tid"))) for e in graph_kernels if "nccl" in e.get("name", "").lower())
    main_stream = comm_streams.most_common(1)[0][0] if comm_streams else None
    ranks = re.findall(r"(?:^|[-_])TP[-_](\d+)(?=[-_.]|$)", path.name, re.I)
    issues = []
    if not kernels:
        issues.append("No GPU kernel events")
    if args.mode == "graph":
        if not launches:
            issues.append("No Graph replay events")
        if not counts or any(n == 0 for n in counts):
            issues.append("Graph replay has missing kernel records")
        if None in correlations or len(valid_ids) != len(correlations):
            issues.append("Missing/duplicate Graph correlation IDs; inspect raw events")
    if args.expect_replays is not None and len(launches) != args.expect_replays:
        issues.append(f"Expected {args.expect_replays} replays, found {len(launches)}")
    if args.expect_kernels_per_replay is not None and (not counts or any(n != args.expect_kernels_per_replay for n in counts)):
        issues.append(f"Per-replay counts do not all match {args.expect_kernels_per_replay}")
    if args.single_stream and len(streams) != 1:
        issues.append(f"Single-stream expectation failed: {dict(streams)}")
    return {
        "file": path.relative_to(relative_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "rank": int(ranks[-1]) if ranks else None,
        "gpu_kernel_events": len(kernels),
        "graph_launch_events": len(launches),
        "graph_kernel_events": len(graph_kernels),
        "kernels_per_replay": counts,
        "count_histogram": dict(collections.Counter(map(str, counts))),
        "graph_kernels_by_stream": dict(streams),
        "main_stream_from_rccl": main_stream,
        "graph_kernel_families": dict(collections.Counter(family(e["name"]) for e in graph_kernels)),
        "graph_kernel_names": dict(collections.Counter(e["name"] for e in graph_kernels)),
        "issues": issues,
    }


def inspect_directory(path, args, output):
    path = Path(path).resolve()
    relative_root = path if path.is_dir() else path.parent
    files = trace_paths(path)
    rows = [inspect_trace(p, relative_root, args) for p in files]
    errors = []
    if not files:
        errors.append("No raw trace files found")
    if args.expect_tp is not None:
        ranks = [r["rank"] for r in rows]
        if len(rows) != args.expect_tp or set(ranks) != set(range(args.expect_tp)):
            errors.append(f"Expected ranks 0..{args.expect_tp-1}, found {ranks}; filename parser may need adapting")
    errors.extend(f"{r['file']}: {issue}" for r in rows for issue in r["issues"])
    result = {
        "schema_version": 1,
        "status": "failed" if errors else "passed",
        "coverage": "matches_explicit_kernel_count" if not errors and args.expect_kernels_per_replay is not None else "not_proven_by_event_presence",
        "expectations": {k: getattr(args, k) for k in ["mode", "expect_tp", "expect_replays", "expect_kernels_per_replay", "single_stream"]},
        "traces": rows,
        "errors": errors,
        "note": "Event presence and stable counts alone do not prove complete coverage. Check model-specific layer/kernel counts. GPU coverage does not prove CPU flow/stack accuracy or physical GPU utilization.",
    }
    save(output, result)
    return result


def api(base, path, payload=None, timeout=600):
    request = urllib.request.Request(base.rstrip("/") + path, data=None if payload is None else json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = {"raw": raw, "status": response.status}
    if isinstance(value, dict) and (value.get("success") is False or value.get("error")):
        raise RuntimeError(f"{path}: {value}")
    return value


def workload(args):
    if args.request_json:
        value = read_json(args.request_json)
        return value.get("request", value)
    if not args.model_path:
        raise ValueError("Provide --request-json or --model-path")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    text = "Explain how a computer processes data and how memory stores information. "
    ids = tokenizer.encode(text * max(200, args.input_tokens), add_special_tokens=False)[:args.input_tokens]
    if len(ids) != args.input_tokens:
        raise ValueError("Could not build the requested input token count")
    return {"input_ids": ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": args.output_tokens, "min_new_tokens": args.output_tokens, "ignore_eos": True}, "stream": False}


def capture(args):
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    sentinels = ["workload.json", "profiler_config.json", "trace_manifest.json", "capture.done", "capture-error.json"]
    if any((root / n).exists() for n in sentinels) or ((root / "trace").exists() and any((root / "trace").iterdir())):
        raise ValueError("Capture output already exists; choose a fresh experiment directory")
    prefix = args.prefix or root.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", prefix):
        raise ValueError("Profile prefix must contain only letters, digits, dot, underscore or hyphen")
    request = workload(args)
    if not isinstance(request, dict) or request.get("stream"):
        raise ValueError("This helper supports one non-streaming request")
    save(root / "workload.json", {"request": request, "warmup_requests": args.warmups, "flush_before_capture": not args.no_flush})
    call = lambda path, payload=None: api(args.base_url, path, payload, args.timeout)
    try:
        save(root / "health-before.json", call("/health"))
        for i in range(args.warmups):
            save(root / f"warmup-{i+1}.json", call("/generate", request))
        if not args.no_flush:
            save(root / "flush-cache.json", call("/flush_cache", {}))
        save(root / "server_info.json", call("/server_info"))
        (root / "trace").mkdir(exist_ok=True)
        config = {"output_dir": str(root / "trace"), "activities": ["CPU", "GPU"], "with_stack": args.with_stack, "record_shapes": True, "profile_by_stage": False, "merge_profiles": False, "profile_prefix": prefix, "profile_id": prefix}
        save(root / "profiler_config.json", config)
        save(root / "profile-start.json", call("/start_profile", config))
        generation_error = None
        stop_error = None
        try:
            begin = time.perf_counter()
            response = call("/generate", request)
            elapsed = time.perf_counter() - begin
            save(root / "profile-request-response.json", response)
            save(root / "request_timing.json", {"profiled_request_seconds": elapsed, "note": "Whole request including prefill, decode, scheduling and profiler overhead; not a decode benchmark."})
            if args.expected_output_tokens is not None and response.get("meta_info", {}).get("completion_tokens") != args.expected_output_tokens:
                raise ValueError("Response token count differs from --expected-output-tokens")
        except Exception as error:
            generation_error = str(error)
        finally:
            try:
                save(root / "profile-stop.json", call("/stop_profile", {}))
            except Exception as error:
                stop_error = str(error)
        if generation_error or stop_error:
            raise RuntimeError(f"generation_error={generation_error}; stop_error={stop_error}")
        save(root / "health-after.json", call("/health"))
        deadline = time.monotonic() + args.flush_wait
        while True:
            files = trace_paths(root)
            if len(files) >= (args.expect_tp or 1):
                try:
                    result = inspect_directory(root, args, root / "trace_manifest.json")
                    break
                except (EOFError, json.JSONDecodeError, OSError):
                    if time.monotonic() >= deadline:
                        raise
            if time.monotonic() >= deadline:
                raise TimeoutError("Trace files did not finish flushing")
            time.sleep(1)
        if result["errors"]:
            raise RuntimeError("; ".join(result["errors"]))
        (root / "capture.done").write_text("success\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "coverage": result["coverage"], "files": len(result["traces"])}))
        return 0
    except Exception as error:
        save(root / "capture-error.json", {"error": str(error), "note": "Do not blindly repeat a timed-out /start_profile. Inspect server profiler state; preserve this attempt and use a fresh directory."})
        raise


def filesystem_path(path):
    path = Path(path).resolve()
    if os.name != "nt" or str(path).startswith("\\\\?\\"):
        return path
    value = str(path)
    return Path("\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value)


def archive_data_filter(member, destination):
    parsed = PurePosixPath(member.name)
    if parsed.as_posix() == ".":
        return None
    if ".." in parsed.parts:
        raise tarfile.FilterError("Parent traversal in archive member")
    member = copy.copy(member)
    member.name = parsed.as_posix()
    member = tarfile.data_filter(member, destination)
    if member is not None and os.name == "nt":
        member.name = os.path.normpath(member.name)
    return member


def download(args):
    if not args.host or args.host.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.@:-]+", args.host):
        raise ValueError("Use a configured SSH alias or host")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.container):
        raise ValueError("Invalid container name")
    remote = PurePosixPath(args.remote_dir)
    if not remote.is_absolute() or remote == PurePosixPath("/") or ".." in remote.parts:
        raise ValueError("Use an absolute, dedicated remote experiment directory")
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "artifacts"
    archive = root / "artifacts.tar.gz"
    if archive.exists() or (destination.exists() and any(destination.iterdir())):
        raise ValueError("Downloaded output exists; choose a fresh destination")
    temporary = root / "artifacts.tar.gz.partial"
    command = shlex.join(["docker", "exec", args.container, "tar", "--exclude=core", "--exclude=core.*", "--exclude=./core", "-czf", "-", "-C", str(remote), "."])
    with temporary.open("wb") as stream:
        subprocess.run([args.ssh_binary, "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3", args.host, command], stdout=stream, timeout=args.timeout, check=True)
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError("Use Python 3.12+ or a Python build with tarfile.data_filter")
    destination.mkdir(exist_ok=True)
    storage = filesystem_path(destination)
    with tarfile.open(temporary, "r:gz") as bundle:
        bundle.extractall(storage, filter=archive_data_filter)
    manifests = list(storage.rglob("trace_manifest.json"))
    if not manifests:
        raise ValueError("No trace_manifest.json to verify download")
    verified = []
    for manifest in manifests:
        data = read_json(manifest)
        rows = data["traces"] if isinstance(data, dict) else data
        if not rows:
            raise ValueError("Empty trace manifest")
        for row in rows:
            file = (manifest.parent / row["file"]).resolve()
            if not file.is_relative_to(storage.resolve()):
                raise ValueError("Manifest path escapes artifacts directory")
            if file.stat().st_size != row["bytes"] or digest(file) != row["sha256"]:
                raise ValueError(f"Checksum mismatch: {file}")
            verified.append(str(file.relative_to(storage)))
    temporary.replace(archive)
    save(root / "download-check.json", {"status": "verified", "files": verified, "remote_host": args.host, "container": args.container, "remote_dir": str(remote)})
    print(json.dumps({"status": "verified", "files": len(verified), "directory": str(destination)}))
    return 0


def expectations(parser):
    parser.add_argument("--mode", choices=["graph", "eager"], default="graph")
    parser.add_argument("--expect-tp", type=int)
    parser.add_argument("--expect-replays", type=int)
    parser.add_argument("--expect-kernels-per-replay", type=int)
    parser.add_argument("--single-stream", action="store_true")


def main():
    parser = argparse.ArgumentParser(description="Capture, inspect and download HCU SGLang per-rank traces")
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser("inspect")
    inspect.add_argument("input")
    inspect.add_argument("--output", required=True)
    expectations(inspect)
    cap = subs.add_parser("capture")
    cap.add_argument("--base-url", default="http://127.0.0.1:30000")
    cap.add_argument("--output-dir", required=True)
    cap.add_argument("--request-json")
    cap.add_argument("--model-path")
    cap.add_argument("--input-tokens", type=int, default=1024)
    cap.add_argument("--output-tokens", type=int, default=32)
    cap.add_argument("--expected-output-tokens", type=int)
    cap.add_argument("--warmups", type=int, default=2)
    cap.add_argument("--prefix")
    cap.add_argument("--timeout", type=float, default=600)
    cap.add_argument("--flush-wait", type=float, default=30)
    cap.add_argument("--no-flush", action="store_true")
    cap.add_argument("--with-stack", action="store_true")
    expectations(cap)
    get = subs.add_parser("download")
    get.add_argument("--host", required=True)
    get.add_argument("--container", required=True)
    get.add_argument("--remote-dir", required=True)
    get.add_argument("--output-dir", required=True)
    get.add_argument("--ssh-binary", default="ssh")
    get.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    for key in ["expect_tp", "expect_replays", "expect_kernels_per_replay", "input_tokens", "output_tokens", "timeout", "flush_wait"]:
        value = getattr(args, key, None)
        if value is not None and value <= 0:
            parser.error(f"{key} must be positive")
    if getattr(args, "warmups", 0) < 0:
        parser.error("warmups cannot be negative")
    try:
        if args.command == "capture":
            return capture(args)
        if args.command == "download":
            return download(args)
        result = inspect_directory(args.input, args, args.output)
        print(json.dumps({"status": result["status"], "coverage": result["coverage"], "errors": result["errors"]}))
        return 1 if result["errors"] else 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
