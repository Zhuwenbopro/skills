#!/usr/bin/env python3
"""Verify a real OpenAI-compatible model endpoint and preserve raw evidence."""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


DEFAULT_CASES = [
    {"name": "arithmetic", "prompt": "Calculate 17 times 19. Reply with the integer only.", "expected": "323"},
    {"name": "capital", "prompt": "What is the capital of France? Reply with the city name only.", "expected": "Paris"},
    {"name": "addition", "prompt": "Calculate 1+1. Reply with the integer only.", "expected": "2"},
]


def write_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def request_json(url, payload=None, timeout=5):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def verify(base_url, model, output, cases=None, wait_seconds=600, request_timeout=300,
           max_tokens=128, pid_file=None, poll_seconds=5):
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    result = {"status": "waiting", "base_url": base_url, "model": model, "tests": []}
    save = lambda: write_result(output, result)
    save()
    try:
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                with urllib.request.urlopen(base_url + "/health", timeout=3) as response:
                    if response.status == 200:
                        result["health"] = 200
                        break
            except (OSError, urllib.error.URLError) as exc:
                result["last_startup_health_error"] = str(exc)
            if pid_file:
                pid = int(Path(pid_file).read_text().strip())
                if pid <= 0:
                    raise RuntimeError("invalid service PID")
                stat = Path(f"/proc/{pid}/stat")
                if not stat.exists() or stat.read_text().split()[2] == "Z":
                    raise RuntimeError("service process exited")
            if time.monotonic() >= deadline:
                raise TimeoutError("model service did not become healthy")
            save()
            time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
        result["models"] = request_json(base_url + "/v1/models")
        ids = [item["id"] for item in result["models"]["data"]]
        if model not in ids:
            raise RuntimeError(f"expected model {model!r}, got {ids!r}")
        result["status"] = "testing"
        save()
        for case in DEFAULT_CASES if cases is None else cases:
            item = {"name": case["name"], "prompt": case["prompt"], "expected": case["expected"]}
            started = time.monotonic()
            try:
                body = request_json(base_url + "/v1/chat/completions", {
                    "model": model, "messages": [{"role": "user", "content": case["prompt"]}],
                    "temperature": 0, "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                }, timeout=request_timeout)
                choice = body["choices"][0]
                answer = (choice["message"].get("content") or "").strip()
                normalize = lambda text: text.strip().casefold().rstrip(".。")
                passed = (normalize(answer) == normalize(str(case["expected"])) and
                          choice["finish_reason"] == "stop" and body.get("model") == model)
                item.update(response=body, seconds=round(time.monotonic() - started, 3), passed=passed)
                print(case["name"], repr(answer), "passed=", passed, flush=True)
            except Exception as exc:
                item.update(error=str(exc), passed=False)
                if isinstance(exc, urllib.error.HTTPError):
                    item["error_body"] = exc.read().decode("utf-8", errors="replace")
            result["tests"].append(item)
            save()
            if not item["passed"]:
                raise RuntimeError(f"generation validation failed: {case['name']}")
        if not result["tests"]:
            raise ValueError("at least one real generation case is required")
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=str(exc))
    save()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", type=Path, help="JSON list of name/prompt/expected cases")
    p.add_argument("--wait-seconds", type=float, default=600)
    p.add_argument("--request-timeout", type=float, default=300)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--pid-file", type=Path, help="Linux container PID file; use only for own service")
    args = p.parse_args()
    if args.wait_seconds < 0 or args.request_timeout <= 0 or args.max_tokens <= 0:
        p.error("invalid time or token limit")
    cases = json.loads(args.cases.read_text(encoding="utf-8-sig")) if args.cases else None
    result = verify(args.base_url, args.model, args.output, cases, args.wait_seconds,
                    args.request_timeout, args.max_tokens, args.pid_file)
    print("VALIDATION_" + result["status"].upper(), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
