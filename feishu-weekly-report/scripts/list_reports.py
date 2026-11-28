"""Read the GPT 工作 index and select archived work reports by date."""

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import date, timedelta
from urllib.parse import urlsplit
from xml.etree import ElementTree


REPORT_TITLE = re.compile(r"(?:(\d{4}-\d{2}-\d{2})\s*[｜|]\s*)?.+?\s*[｜|]\s*工作汇报")
DOCUMENT_HOST = re.compile(r"(?:[a-z0-9-]+\.)*(?:feishu\.cn|larksuite\.com|doubao\.com)")
DOCUMENT_PATH = re.compile(r"/(?:docx|wiki)/[A-Za-z0-9_-]+/?")


def normalize_document_url(url):
    target = urlsplit(url.strip())
    host = target.netloc.lower()
    if (
        target.scheme != "https"
        or not DOCUMENT_HOST.fullmatch(host)
        or not DOCUMENT_PATH.fullmatch(target.path)
    ):
        raise ValueError("Expected an HTTPS Feishu or Lark Docx/Wiki URL")
    return f"https://{host}{target.path.rstrip('/')}"


def fetch_index(index_url):
    cli = shutil.which("lark-cli")
    if not cli:
        raise ValueError("lark-cli was not found on PATH")
    result = subprocess.run(
        [cli, "docs", "+fetch", "--doc", index_url, "--as", "user",
         "--doc-format", "xml", "--format", "json"],
        capture_output=True,
        encoding="utf-8",
        timeout=60,
        env=os.environ | {
            "LARKSUITE_CLI_NO_UPDATE_NOTIFIER": "1",
            "LARKSUITE_CLI_NO_SKILLS_NOTIFIER": "1",
        },
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"lark-cli exited with {result.returncode}")
    payload = json.loads(result.stdout)
    if payload.get("ok") is not True:
        raise ValueError("lark-cli did not return a successful index response")
    return payload["data"]["document"]["content"]


def select_reports(content, start, end, index_url):
    reports = {}
    undated = {}
    dated_urls = set()
    root = ElementTree.fromstring(f"<document>{content}</document>")
    for anchor in root.iter("a"):
        title = " ".join("".join(anchor.itertext()).split())
        match = REPORT_TITLE.fullmatch(title)
        if not match:
            continue
        try:
            url = normalize_document_url(anchor.get("href", ""))
        except ValueError:
            continue
        if url == index_url:
            continue

        try:
            report_date = date.fromisoformat(match[1]) if match[1] else None
        except ValueError:
            report_date = None
        if report_date is None:
            undated.setdefault(url, {"title": title, "url": url})
            continue
        dated_urls.add(url)
        if start <= report_date <= end:
            reports.setdefault(url, {"date": report_date.isoformat(), "title": title, "url": url})

    return {
        "index": index_url,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "reports": sorted(reports.values(), key=lambda report: report["date"]),
        "undated": [report for url, report in undated.items() if url not in dated_urls],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", help="Feishu report index URL, defaults to FEISHU_REPORT_INDEX_URL")
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="Inclusive end date in Asia/Shanghai")
    parser.add_argument("--start", type=date.fromisoformat, help="Inclusive start date, defaults to end minus six days")
    args = parser.parse_args()
    index = args.index if args.index is not None else os.environ.get("FEISHU_REPORT_INDEX_URL", "")
    if not index.strip():
        parser.error("provide --index or set FEISHU_REPORT_INDEX_URL")
    try:
        index_url = normalize_document_url(index)
    except ValueError as error:
        parser.error(str(error))

    start = args.start or args.end - timedelta(days=6)
    if start > args.end:
        parser.error("--start must be on or before --end")

    try:
        manifest = select_reports(fetch_index(index_url), start, args.end, index_url)
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, ElementTree.ParseError) as error:
        parser.exit(1, f"Could not read the Feishu report index: {error}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
