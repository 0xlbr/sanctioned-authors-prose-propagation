#!/usr/bin/env python3
"""Method 1 query runner for the Sanctioned Prose Benchmark (search on, 8k, effort high)."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

def load_env_file() -> None:
    for env_path in (HERE / ".env", Path.cwd() / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

QUESTIONS = Path(os.environ.get("SPB_QUESTIONS", HERE / "method1_questions_200.csv"))
OUT_ROOT = Path(os.environ.get("SPB_OUT", Path.cwd() / "answers"))
UA = "random20k-benchmark/1.0 (anomalyai-research)"

ARMS = [
    {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "env_key": "ANTHROPIC_API_KEY",
    },
    {
        "provider": "openai",
        "model": "gpt-5.6-terra",
        "env_key": "OPENAI_API_KEY",
    },
    {
        "provider": "xai",
        "model": "grok-4.3",
        "env_key": "XAI_API_KEY",
    },
]

WIKI_RE = re.compile(r"https?://(?:[a-z]+\.)?wikipedia\.org/wiki/[^\s\)\]\"']+")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def post_json(url: str, body: dict, headers: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def parse_anthropic(raw: dict) -> dict:
    content = raw.get("content") or []
    texts = []
    queries = []
    results = []
    rank = 0
    search_calls = 0
    for block in content:
        btype = block.get("type")
        if btype == "text":
            texts.append(block.get("text") or "")
        elif btype == "server_tool_use" and block.get("name") == "web_search":
            search_calls += 1
            inp = block.get("input") or {}
            q = inp.get("query")
            if q:
                queries.append({"query": q})
        elif btype == "web_search_tool_result":
            for item in block.get("content") or []:
                if item.get("type") != "web_search_result":
                    continue
                rank += 1
                results.append(
                    {
                        "rank": rank,
                        "url": item.get("url"),
                        "title": item.get("title"),
                    }
                )
    answer_text = "".join(texts).strip()
    cited_urls = {m.group(0).rstrip(".,;") for m in WIKI_RE.finditer(answer_text)}
    citations = []
    for r in results:
        url = r.get("url") or ""
        if url in cited_urls or "wikipedia.org" in url:
            citations.append({"url": url, "title": r.get("title")})
    if not citations:
        for url in sorted(cited_urls):
            citations.append({"url": url, "title": ""})
    usage = raw.get("usage") or {}
    return {
        "answer_text": answer_text,
        "queries": queries,
        "results": results,
        "citations": citations[:20],
        "search_calls": search_calls,
        "stop_reason": raw.get("stop_reason"),
        "model_echoed": raw.get("model"),
        "usage": {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        },
        "raw": [raw],
    }


def parse_openai(raw: dict) -> dict:
    texts = []
    queries = []
    results = []
    rank = 0
    search_calls = 0
    for block in raw.get("output") or []:
        btype = block.get("type")
        if btype == "message":
            for part in block.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(part.get("text") or "")
        elif btype == "web_search_call":
            search_calls += 1
            action = block.get("action") or {}
            for q in action.get("queries") or []:
                queries.append({"query": q})
            if action.get("query") and not action.get("queries"):
                queries.append({"query": action["query"]})
            for src in action.get("sources") or []:
                rank += 1
                results.append(
                    {
                        "rank": rank,
                        "url": src.get("url"),
                        "title": src.get("title"),
                    }
                )
    answer_text = "".join(texts).strip()
    cited = {m.group(0).rstrip(".,;") for m in WIKI_RE.finditer(answer_text)}
    citations = [{"url": u, "title": ""} for u in sorted(cited)]
    usage = raw.get("usage") or {}
    return {
        "answer_text": answer_text,
        "queries": queries,
        "results": results,
        "citations": citations[:20],
        "search_calls": search_calls,
        "stop_reason": raw.get("status"),
        "model_echoed": raw.get("model"),
        "usage": usage,
        "raw": [raw],
    }


def call_anthropic(question: str, model: str, api_key: str) -> dict:
    body = {
        "model": model,
        "max_tokens": 8000,
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "high"},
        "messages": [{"role": "user", "content": question}],
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
                "user_location": {
                    "type": "approximate",
                    "city": "London",
                    "country": "GB",
                    "region": "England",
                    "timezone": "Europe/London",
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "web_search"},
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    raw = post_json("https://api.anthropic.com/v1/messages", body, headers)
    parsed = parse_anthropic(raw)
    parsed["provider"] = "anthropic"
    parsed["model"] = model
    return parsed


def call_openai(question: str, model: str, api_key: str) -> dict:
    body = {
        "model": model,
        "input": question,
        "max_output_tokens": 8000,
        "reasoning": {"effort": "high"},
        "tools": [{"type": "web_search"}],
        "tool_choice": {"type": "web_search"},
        "max_tool_calls": 5,
        "include": ["web_search_call.action.sources"],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    raw = post_json("https://api.openai.com/v1/responses", body, headers)
    parsed = parse_openai(raw)
    parsed["provider"] = "openai"
    parsed["model"] = model
    return parsed


def call_xai(question: str, model: str, api_key: str) -> dict:
    """xAI Agent Tools API."""
    body = {
        "model": model,
        "input": question,
        "max_output_tokens": 8000,
        "temperature": 0,
        "reasoning": {"effort": "high"},
        "tools": [
            {
                "type": "web_search",
                "user_location": {
                    "type": "approximate",
                    "city": "London",
                    "country": "GB",
                    "timezone": "Europe/London",
                },
            }
        ],
        "tool_choice": "required",
        "include": ["web_search_call.action.sources"],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    raw = post_json("https://api.x.ai/v1/responses", body, headers)
    parsed = parse_openai(raw)
    parsed["provider"] = "xai"
    parsed["model"] = model
    return parsed


def load_questions() -> list[dict]:
    with QUESTIONS.open(newline="") as fh:
        return list(csv.DictReader(fh))


def record_is_complete(rec: dict) -> bool:
    if rec.get("error"):
        return False
    return bool((rec.get("answer_text") or "").strip())


def validate_answer(rec: dict) -> tuple[bool, str]:
    if rec.get("error"):
        return False, str(rec["error"])[:160]
    ans = (rec.get("answer_text") or "").strip()
    if len(ans) < 40:
        return False, f"answer too short ({len(ans)} chars)"
    if rec.get("stop_reason") == "error":
        return False, "stop_reason=error"
    return True, f"ok ({len(ans)} chars, searches={rec.get('search_calls', 0)})"


def write_deliverable_csv(provider_dir: Path, records: list[dict], model: str) -> None:
    out = OUT_ROOT / f"method 1 - generated questions - {model} answers.csv"
    fields = [
        "n",
        "article",
        "author_class",
        "tier",
        "position_band",
        "position_frac",
        "question",
        "provider",
        "model",
        "model_echoed",
        "search_calls",
        "queries",
        "results",
        "citations",
        "stop_reason",
        "error",
        "answer_text",
    ]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            row["queries"] = json.dumps(r.get("queries") or [])
            row["results"] = len(r.get("results") or [])
            row["citations"] = len(r.get("citations") or [])
            w.writerow(row)


def run_arm(arm: dict, rows: list[dict], *, limit: int = 0) -> int:
    api_key = os.environ.get(arm["env_key"])
    if not api_key:
        log(f"SKIP {arm['provider']}: {arm['env_key']} not set")
        return 0

    if limit:
        rows = rows[:limit]

    provider_dir = OUT_ROOT / arm["provider"]
    raw_dir = provider_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = []
    errors = 0

    for row in rows:
        n = int(row["n"])
        raw_path = raw_dir / f"q{n:03d}.json"
        if raw_path.exists():
            rec = json.loads(raw_path.read_text())
            if record_is_complete(rec):
                rec.setdefault("model", arm["model"])
                rec.setdefault("provider", arm["provider"])
                records.append(rec)
                continue
            log(f"  q{n} retrying (incomplete cached record)")

        question = row["question"]
        parsed = None
        for attempt in range(5):
            try:
                if arm["provider"] == "anthropic":
                    parsed = call_anthropic(question, arm["model"], api_key)
                elif arm["provider"] == "xai":
                    parsed = call_xai(question, arm["model"], api_key)
                else:
                    parsed = call_openai(question, arm["model"], api_key)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode()[:500]
                if e.code in (429, 529) and attempt < 4:
                    wait = 15 * (attempt + 1)
                    log(f"  q{n} HTTP {e.code}, sleep {wait}s")
                    time.sleep(wait)
                    continue
                parsed = {
                    "answer_text": "",
                    "queries": [],
                    "results": [],
                    "citations": [],
                    "search_calls": 0,
                    "stop_reason": "error",
                    "model_echoed": arm["model"],
                    "model": arm["model"],
                    "provider": arm["provider"],
                    "usage": {},
                    "raw": [],
                    "error": f"HTTP {e.code}: {detail}",
                }
                errors += 1
                break
            except Exception as e:  # noqa: BLE001
                if attempt < 4:
                    time.sleep(10 * (attempt + 1))
                    continue
                parsed = {
                    "answer_text": "",
                    "queries": [],
                    "results": [],
                    "citations": [],
                    "search_calls": 0,
                    "stop_reason": "error",
                    "model_echoed": arm["model"],
                    "model": arm["model"],
                    "provider": arm["provider"],
                    "usage": {},
                    "raw": [],
                    "error": str(e),
                }
                errors += 1
                break

        rec = {
            "n": n,
            "article": row["article"],
            "question": question,
            "author_class": row.get("author_class", ""),
            "tier": row.get("tier", ""),
            "position_band": row.get("position_band", ""),
            "position_frac": row.get("position_frac", ""),
            "coverage": row.get("coverage", ""),
            "pinned": row.get("pinned", ""),
            "source_type": row.get("source_type", ""),
            **{k: parsed[k] for k in parsed if k != "raw"},
            "resumes": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if parsed.get("error"):
            rec["error"] = parsed["error"]
        rec["raw"] = parsed.get("raw") or []
        raw_path.write_text(json.dumps(rec, indent=2))
        records.append(rec)
        log(
            f"  {arm['provider']} q{n}/200 "
            f"searches={rec.get('search_calls', 0)} "
            f"chars={len(rec.get('answer_text') or '')}"
        )
        time.sleep(1)

    records.sort(key=lambda r: r["n"])
    provider_dir.joinpath("records.json").write_text(json.dumps(records, indent=2))
    write_deliverable_csv(provider_dir, records, arm["model"])
    log(f"{arm['provider']} done: {len(records)} records, {errors} errors")
    return errors


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Method 1 LLM answers (random20k)")
    parser.add_argument(
        "--providers",
        default="anthropic,openai,xai",
        help="comma-separated: anthropic, openai, xai",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only run first N questions (0 = all)",
    )
    parser.add_argument(
        "--probe",
        type=int,
        default=0,
        help="run first N questions and validate; exit 1 if any bad",
    )
    parser.add_argument(
        "--openai-model",
        default="",
        help="override OpenAI model slug (e.g. gpt-5.6-sol)",
    )
    args = parser.parse_args()

    load_env_file()
    if not QUESTIONS.exists():
        log(f"ERROR: missing {QUESTIONS}")
        return 1

    wanted = {p.strip() for p in args.providers.split(",") if p.strip()}
    arms = [a for a in ARMS if a["provider"] in wanted]
    if args.openai_model:
        arms = [
            {**arm, "model": args.openai_model} if arm["provider"] == "openai" else arm
            for arm in arms
        ]
    if not arms:
        log("ERROR: no matching providers")
        return 1

    if not any(os.environ.get(a["env_key"]) for a in arms):
        log("SKIPPED: set API keys in .env for selected providers")
        return 0

    rows = load_questions()
    limit = args.probe or args.limit or 0
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"running Method 1 answers for {len(rows)} questions (limit={limit or 'all'})")
    total_errors = 0
    for arm in arms:
        total_errors += run_arm(arm, rows, limit=limit)

    if args.probe:
        probed = rows[: args.probe]
        bad = []
        for row in probed:
            n = int(row["n"])
            for arm in arms:
                raw_path = OUT_ROOT / arm["provider"] / "raw" / f"q{n:03d}.json"
                if not raw_path.exists():
                    bad.append((n, arm["provider"], "missing raw file"))
                    continue
                rec = json.loads(raw_path.read_text())
                ok, msg = validate_answer(rec)
                log(f"PROBE q{n} {arm['provider']}: {msg}")
                if not ok:
                    bad.append((n, arm["provider"], msg))
        if bad:
            log(f"PROBE FAILED: {len(bad)} bad of {args.probe * len(arms)} checks")
            return 1
        log(f"PROBE OK: {args.probe} question(s) per provider look well-formed")
        return 0

    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
