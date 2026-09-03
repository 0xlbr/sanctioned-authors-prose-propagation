"""Downstream measurement — canonical implementations of DEFINITIONS.md
§ "Downstream measurement".

Tokenisation and span matching are byte-identical to the frozen pipeline
(llm_overlap_blocked_attribution.py: tok, MIN_SPAN, find_spans_with_owners).
"""
from __future__ import annotations

import re
from collections import defaultdict

MIN_VERBATIM_SPAN = 8

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tok(s: str) -> list[str]:
    """Canonical tokeniser for verbatim-span matching: lowercase [a-z0-9']
    runs. A **prose token** is one element of this list after the prose
    filter has stripped templates/refs/link-targets/HTML."""
    return _TOKEN_RE.findall(s.lower())


def normalize_match_tokens(tokens: list[str]) -> list[str]:
    """Relax token streams before verbatim matching (benchmark / QA).

    Keeps the ≥8-word rule but tolerates punctuation drift between answer
    and corpus: wiki quote tokens, hyphen splits, and digit-group commas.
    Case is already folded by ``tok()``.
    """
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t or set(t) <= {"'"} or t == '"':
            i += 1
            continue
        if (
            t == "-"
            and out
            and i + 1 < len(tokens)
            and out[-1][0].isalnum()
            and tokens[i + 1][0].isalnum()
        ):
            out[-1] = out[-1] + "-" + tokens[i + 1]
            i += 2
            continue
        if (
            t.isdigit()
            and i + 2 < len(tokens)
            and tokens[i + 1] == ","
            and tokens[i + 2].isdigit()
        ):
            merged = t + tokens[i + 2]
            i += 3
            while i + 2 <= len(tokens) and tokens[i] == "," and tokens[i + 1].isdigit():
                merged += tokens[i + 1]
                i += 2
            out.append(merged)
            continue
        out.append(t)
        i += 1
    return out


def match_tok(s: str) -> list[str]:
    """``tok()`` followed by ``normalize_match_tokens()`` for relaxed matching."""
    return normalize_match_tokens(tok(s))


def find_verbatim_spans(
    answer: str,
    corpus_tokens: list[str],
    owners: list[str | None] | None = None,
    *,
    relaxed: bool = False,
) -> list[tuple[int, str | None]]:
    """Term: **Verbatim span** — >=8 consecutive identical tokenised words
    shared between a model answer and the corpus prose stream. Paraphrase
    does not count.

    Returns [(span_length_tokens, owner_of_span_start), ...]. `owners` maps
    each corpus token to the account that introduced it (author_passages);
    pass None if attribution is not needed.
    """
    aw = match_tok(answer) if relaxed else tok(answer)
    cw_full = normalize_match_tokens(corpus_tokens) if relaxed else corpus_tokens
    n = MIN_VERBATIM_SPAN
    if len(aw) < n or len(cw_full) < n:
        return []
    own = owners if owners is not None else []
    starts: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for j in range(len(cw_full) - n + 1):
        starts[tuple(cw_full[j : j + n])].append(j)
    hits: list[tuple[int, str | None]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(len(aw) - n + 1):
        for j in starts.get(tuple(aw[i : i + n]), []):
            k = n
            while (
                i + k < len(aw)
                and j + k < len(cw_full)
                and aw[i + k] == cw_full[j + k]
            ):
                k += 1
            key = (i, i + k)
            if key in seen or k < n:
                continue
            seen.add(key)
            owner = own[j] if j < len(own) else None
            hits.append((k, owner))
    return hits
