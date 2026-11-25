"""Run the official SGLang serving benchmark once per concurrency level."""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

REFERENCE_CONCURRENCY = list(range(1, 9)) + list(range(10, 33, 2)) + [40]
METRICS = {
    "duration_s": "duration",
    "rps": "request_throughput",
    "input_throughput_tok_s": "input_throughput",
    "generate_throughput_tok_s": "output_throughput",
    "total_throughput_tok_s": "total_throughput",
    "peak_output_throughput": "max_output_tokens_per_s",
    "peak_concurrency": "max_concurrent_requests",
}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def concurrency_list(value):
    levels = [positive_int(item) for item in value.split(",")]
    if len(levels) != len(set(levels)):
        raise argparse.ArgumentTypeError("concurrency levels must be unique")
    return levels


def read_result(path):
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        results = [json.loads(line) for line in stream if line.strip()]
    if len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError("expected exactly one JSONL result object")
    return results[0]


def run_case(args, concurrency, prompts):
    directory = args.output_dir / f"c{concurrency}"
    directory.mkdir()
    result_path = directory / "result.jsonl"
    seed = args.seed + concurrency
    command = [
        sys.executable, "-m", args.module,
        "--backend", "sglang", "--base-url", args.base_url,
        "--model", args.model, "--tokenizer", args.tokenizer or args.model,
        "--dataset-name", "random-ids", "--tokenize-prompt",
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--random-range-ratio", "1", "--seed", str(seed),
        "--num-prompts", str(prompts),
        "--max-concurrency", str(concurrency), "--request-rate", "inf",
        "--warmup-requests", str(args.warmup_requests),
        "--output-file", str(result_path),
    ]
    (directory / "command.json").write_text(json.dumps(command) + "\n", encoding="utf-8")

    status = "FAIL"
    exit_code = None
    error = ""
    native = {}
    with (directory / "console.log").open("wb") as log:
        try:
            # SGLang's CI mode can flush the server cache even without a CLI flag.
            environment = os.environ.copy()
            environment.pop("SGLANG_IS_IN_CI", None)
            exit_code = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT,
                timeout=args.timeout, env=environment,
            ).returncode
            native = read_result(result_path)
            if exit_code != 0:
                error = f"benchmark exited with code {exit_code}"
            elif not native:
                error = "missing benchmark result"
            elif native.get("completed") != prompts:
                error = f"completed {native.get('completed')} of {prompts} requests"
            elif native.get("total_input_tokens") != prompts * args.input_len:
                error = "unexpected input token count"
            elif native.get("total_output_tokens", native.get("total_generated_tokens")) != prompts * args.output_len:
                error = "unexpected output token count"
            else:
                status = "PASS"
        except subprocess.TimeoutExpired:
            error = f"benchmark timed out after {args.timeout} seconds"
        except (OSError, ValueError) as exc:
            error = str(exc)

    row = {
        "case_name": "4k_1k" if (args.input_len, args.output_len) == (4096, 1024)
        else f"{args.input_len}in_{args.output_len}out",
        "mode": args.mode,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "tokens_per_request": args.input_len + args.output_len,
        "dataset_name": "random-ids",
        "random_range_ratio": 1,
        "request_rate": "inf",
        "seed": seed,
        "num_prompts": prompts,
        "max_concurrency": concurrency,
        "case_status": status,
        "case_exit_code": exit_code,
        "benchmark_result_found": int(bool(native)),
        "case_error": error,
    }
    for label, source in METRICS.items():
        row[label] = native.get(source, native.get(label))
    for name, value in native.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            row.setdefault(name, value)
    return row


def save_summary(directory, rows):
    columns = list(dict.fromkeys(name for row in rows for name in row))
    temporary = directory / "summary.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(directory / "summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--module", choices=["sglang.benchmark.serving", "sglang.bench_serving"],
                        default="sglang.benchmark.serving")
    parser.add_argument("--mode", choices=["burst", "steady"], default="burst")
    parser.add_argument("--concurrency", type=concurrency_list, default=REFERENCE_CONCURRENCY)
    parser.add_argument("--input-len", type=positive_int, default=4096)
    parser.add_argument("--output-len", type=positive_int, default=1024)
    parser.add_argument("--prompt-multiplier", type=positive_int)
    parser.add_argument("--warmup-requests", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="base seed; add concurrency for each case")
    parser.add_argument("--timeout", type=positive_int, default=1800, help="per-case timeout in seconds")
    args = parser.parse_args()
    try:
        endpoint = urlsplit(args.base_url)
    except ValueError as exc:
        parser.error(f"invalid --base-url: {exc}")
    if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname
            or endpoint.username is not None or endpoint.query or endpoint.fragment):
        parser.error("--base-url must be an HTTP(S) endpoint without credentials, query, or fragment")
    if not 0 <= args.seed <= 2**32 - 1 - max(args.concurrency):
        parser.error("--seed plus concurrency must fit in an unsigned 32-bit integer")
    if args.mode == "burst" and args.prompt_multiplier is not None:
        parser.error("--prompt-multiplier is for steady mode")
    if args.mode == "steady" and args.prompt_multiplier is not None and args.prompt_multiplier < 2:
        parser.error("steady mode requires a prompt multiplier of at least 2")
    if args.output_dir.exists():
        parser.error("--output-dir must be a new path")

    args.output_dir.mkdir(parents=True)
    multiplier = 1 if args.mode == "burst" else (args.prompt_multiplier or 5)
    rows = []
    for concurrency in args.concurrency:
        row = run_case(args, concurrency, concurrency * multiplier)
        rows.append(row)
        save_summary(args.output_dir, rows)
        print(f"concurrency={concurrency}: {row['case_status']} {row['case_error']}", flush=True)
        if row["case_status"] != "PASS":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
