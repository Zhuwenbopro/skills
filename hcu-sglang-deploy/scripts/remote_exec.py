#!/usr/bin/env python3
"""Stream a UTF-8 shell script through existing SSH configuration."""
import argparse
import base64
from pathlib import Path
import re
import shlex
import subprocess
import sys


def build_command(host, script, container=None, workdir="/home"):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*", host):
        raise ValueError("Use an SSH alias or user@host, without shell options")
    script = script.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    target = "bash -s"
    if container:
        target = "docker exec -i -w {} {} bash -s".format(shlex.quote(workdir), shlex.quote(container))
    remote = "printf %s {} | base64 -d | {}".format(shlex.quote(encoded), target)
    return ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=3", host, remote]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True)
    p.add_argument("--container")
    p.add_argument("--workdir", default="/home")
    p.add_argument("--file", type=Path, help="UTF-8 shell file; otherwise read stdin")
    p.add_argument("--timeout", type=float, default=0, help="Local SSH limit in seconds; 0 means unlimited")
    args = p.parse_args()
    if args.timeout < 0:
        p.error("timeout cannot be negative")
    script = args.file.read_text(encoding="utf-8-sig") if args.file else sys.stdin.read()
    if not script.strip():
        p.error("empty script")
    try:
        return subprocess.run(build_command(args.host, script, args.container, args.workdir),
                              timeout=args.timeout or None).returncode
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired:
        print("SSH timed out; inspect remote processes before retrying mutations", file=sys.stderr)
        return 124


if __name__ == "__main__":
    sys.exit(main())
