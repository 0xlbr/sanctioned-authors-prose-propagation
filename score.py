#!/usr/bin/env python3
"""Score Method 1 answers against the frozen v2_eb corpus.

Canonical scoring: strict tok(), >=8 consecutive identical words.
Scoring primitives ship in downstream.py in this pack.

The published statistic is the paste-rate contrast between the two
question arms: questions written from an ever-blocked editor's passage
(author_class=anomalous, 87 questions) versus questions written from a
never-blocked editor's passage (author_class=normal, 113 questions).

  python3 score.py --answers ./answers
  python3 score.py --answers ./reference_answers
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODELS = {
    "anthropic": ("claude-sonnet-4-6", "method 1 - generated questions - claude-sonnet-4-6 answers.csv"),
    "xai": ("grok-4.3", "method 1 - generated questions - grok-4.3 answers.csv"),
    "gpt": ("gpt-5.6-terra", "method 1 - generated questions - gpt-5.6-terra answers.csv"),
}

tok = None
MIN_SPAN = None


def _bind():
    global tok, MIN_SPAN
    import importlib.util

    path = HERE / "downstream.py"
    spec = importlib.util.spec_from_file_location("spb_downstream", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tok = mod.tok
    MIN_SPAN = mod.MIN_VERBATIM_SPAN


def load_corpus(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    slugs, toks = [], []
    for art in payload["articles"]:
        slugs.append(art["slug"])
        toks.append(art["tokens"])
    return slugs, toks


def has_paste(answer: str, articles_tok: list[list[str]], grams_index: dict) -> bool:
    """Does the answer share a >=8-word verbatim run with any corpus article?"""
    aw = tok(answer)
    n = MIN_SPAN
    if len(aw) < n:
        return False
    return any(tuple(aw[i : i + n]) in grams_index for i in range(len(aw) - n + 1))


def build_gram_index(articles_tok: list[list[str]]) -> dict:
    n = MIN_SPAN
    index = {}
    for cw in articles_tok:
        for j in range(len(cw) - n + 1):
            index[tuple(cw[j : j + n])] = True
    return index


def load_questions(path: Path) -> dict[str, str]:
    """question n -> author_class (anomalous | normal)."""
    return {r["n"]: r["author_class"] for r in csv.DictReader(path.open(newline=""))}


def load_answer_csv(path: Path) -> list[dict]:
    return [r for r in csv.DictReader(path.open(newline="")) if (r.get("answer_text") or "").strip()]


def find_csvs(answers: Path) -> list[tuple[str, Path]]:
    if answers.is_file():
        return [(answers.stem, answers)]
    found = []
    for _key, (model, name) in MODELS.items():
        path = answers / name
        if path.is_file():
            found.append((model, path))
    if not found:
        found = [(p.stem, p) for p in sorted(answers.glob("*.csv"))]
    return found


def arm_stats(rows: list[tuple[str, bool]]) -> dict:
    """rows: (author_class, pasted). Paste counts per arm and the paste-rate ratio."""
    out = {}
    for cls in ("anomalous", "normal"):
        sub = [p for c, p in rows if c == cls]
        out[cls] = {"n": len(sub), "paste": sum(sub)}
    a, b = out["anomalous"], out["normal"]
    ra = a["paste"] / a["n"] if a["n"] else None
    rb = b["paste"] / b["n"] if b["n"] else None
    out["paste_rate_ratio"] = round(ra / rb, 2) if ra and rb else None
    return out


def bootstrap_ratio_ci(by_question: dict[str, list[bool]], classes: dict[str, str],
                       *, n_boot: int = 10_000, seed: int = 42, alpha: float = 0.05):
    """Clustered bootstrap over questions: resample question ids with
    replacement within each arm, carrying every answer to that question."""
    qa = [n for n in by_question if classes.get(n) == "anomalous"]
    qn = [n for n in by_question if classes.get(n) == "normal"]
    if not qa or not qn:
        return None
    rng = random.Random(seed)
    ratios = []
    for _ in range(n_boot):
        ea = [p for n in rng.choices(qa, k=len(qa)) for p in by_question[n]]
        en = [p for n in rng.choices(qn, k=len(qn)) for p in by_question[n]]
        rn = sum(en) / len(en)
        if rn == 0:
            continue
        ratios.append((sum(ea) / len(ea)) / rn)
    ratios.sort()
    lo = ratios[int((alpha / 2) * len(ratios))]
    hi = ratios[min(len(ratios) - 1, int((1 - alpha / 2) * len(ratios)))]
    return round(lo, 2), round(hi, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanctioned Prose Benchmark - Method 1 scorer")
    parser.add_argument("--answers", type=Path, default=HERE / "answers",
                        help="directory of Method 1 CSVs, or a single CSV")
    parser.add_argument("--corpus", type=Path, default=HERE / "corpus_v2_eb.json.gz")
    parser.add_argument("--questions", type=Path, default=HERE / "method1_questions_200.csv")
    args = parser.parse_args()
    _bind()
    for p, label in ((args.corpus, "corpus"), (args.answers, "answers"), (args.questions, "questions")):
        if not p.exists():
            print(f"ERROR: missing {label} {p}", file=sys.stderr)
            return 1

    slugs, toks = load_corpus(args.corpus)
    classes = load_questions(args.questions)
    print(f"corpus {len(slugs)} articles · {sum(len(t) for t in toks):,} tokens")
    print("matching strict tok() · >=8 words\n")

    csvs = find_csvs(args.answers)
    if not csvs:
        print("ERROR: no answer CSVs found", file=sys.stderr)
        return 1

    grams = build_gram_index(toks)
    print(f"{'model':<22} {'n':>4} {'paste':>10} {'blocked-author':>16} {'other':>14} {'pr-ratio':>8}")
    all_rows: list[tuple[str, bool]] = []
    by_question: dict[str, list[bool]] = defaultdict(list)
    pooled_n = pooled_paste = 0
    for model, path in csvs:
        rows = load_answer_csv(path)
        marks = []
        for r in rows:
            cls = classes.get(str(r.get("n")), "normal")
            pasted = has_paste(r["answer_text"], toks, grams)
            marks.append((cls, pasted))
            by_question[str(r.get("n"))].append(pasted)
        all_rows.extend(marks)
        pooled_n += len(marks)
        pooled_paste += sum(p for _c, p in marks)
        st = arm_stats(marks)
        a, b = st["anomalous"], st["normal"]
        print(f"{model:<22} {len(marks):>4} {100*sum(p for _c,p in marks)/len(marks):>9.1f}% "
              f"{a['paste']:>4}/{a['n']:<3} {100*a['paste']/a['n'] if a['n'] else 0:>5.1f}% "
              f"{b['paste']:>4}/{b['n']:<3} {100*b['paste']/b['n'] if b['n'] else 0:>5.1f}% "
              f"{st['paste_rate_ratio'] if st['paste_rate_ratio'] is not None else '—':>7}×")

    if len(csvs) > 1:
        st = arm_stats(all_rows)
        a, b = st["anomalous"], st["normal"]
        ci = bootstrap_ratio_ci(by_question, classes)
        ci_txt = f"   95% CI [{ci[0]}, {ci[1]}]" if ci else ""
        print(f"{'pooled':<22} {pooled_n:>4} {100*pooled_paste/pooled_n:>9.1f}% "
              f"{a['paste']:>4}/{a['n']:<3} {100*a['paste']/a['n']:>5.1f}% "
              f"{b['paste']:>4}/{b['n']:<3} {100*b['paste']/b['n']:>5.1f}% "
              f"{st['paste_rate_ratio']:>7}×{ci_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
