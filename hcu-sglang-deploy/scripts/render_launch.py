#!/usr/bin/env python3
"""Render a JSON profile to a launch script; performs no remote actions."""
import argparse
import json
from pathlib import Path
import re
import shlex


OWNED_FLAGS = {"--model-path", "--served-model-name", "--host", "--port", "--tp-size",
               "--tp", "--tensor-parallel-size", "--base-gpu-id", "--gpu-id-step"}
DEVICE_KEYS = {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}


def render(profile):
    devices = profile["devices"]
    if (not isinstance(devices, list) or not devices or
            any(type(d) is not int or d < 0 for d in devices) or len(set(devices)) != len(devices)):
        raise ValueError("devices must contain unique nonnegative physical IDs")
    port = profile.get("port", 30000)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("invalid TCP port")
    for key in ("model_path", "python", "workdir"):
        value = profile[key]
        if not isinstance(value, str) or not value.startswith("/") or "\n" in value:
            raise ValueError(f"{key} must be an absolute container path")
    paths = profile["pythonpath"]
    if not isinstance(paths, list) or not paths or any(not isinstance(x, str) or not x.startswith("/") or ":" in x or "\n" in x for x in paths):
        raise ValueError("pythonpath must contain absolute POSIX directories")
    extra = profile.get("launch_args", [])
    if not isinstance(extra, list) or any(not isinstance(x, str) for x in extra):
        raise ValueError("launch_args must be a string list")
    if any(x.split("=", 1)[0] in OWNED_FLAGS for x in extra):
        raise ValueError("launch_args cannot override profile-owned model, TP, device or port flags")
    env = profile.get("env", {})
    if not isinstance(env, dict):
        raise ValueError("env must be an object")
    for key, value in env.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in DEVICE_KEYS or key == "PYTHONPATH":
            raise ValueError(f"invalid or conflicting environment key: {key}")
        if not isinstance(value, str):
            raise ValueError("environment values must be strings")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "cd " + shlex.quote(profile["workdir"]),
             "unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL",
             "export HIP_VISIBLE_DEVICES=" + shlex.quote(",".join(map(str, devices))),
             "export PYTHONPATH=" + shlex.quote(":".join(paths))]
    lines += ["export {}={}".format(k, shlex.quote(v)) for k, v in env.items()]
    command = [profile["python"], "-u", "-m", "hcu_sglang", "--model-path", profile["model_path"],
               "--served-model-name", profile["served_model_name"], "--host", profile.get("bind_host", "127.0.0.1"),
               "--port", str(port), "--tp-size", str(len(devices)), *extra]
    if any(not isinstance(x, str) or "\x00" in x for x in command):
        raise ValueError("invalid command argument")
    return "\n".join([*lines, "exec " + shlex.join(command)]) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        content = render(json.loads(args.profile.read_text(encoding="utf-8-sig")))
    except (ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)


if __name__ == "__main__":
    main()
