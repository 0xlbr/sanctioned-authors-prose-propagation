# Sanctioned Authors' Prose Propagation

Search-enabled language models sometimes copy Wikipedia word for word. This benchmark measures two things: **how often they copy**, and **whether questions about text written by editors Wikipedia has blocked elicit that copying more often**.

It is not a capability leaderboard. One topic mix, one day, retrieval (search on), not memorisation.

This pack is **Sanctioned Prose Benchmark v2_eb** — a general-English-Wikipedia instrument.

**Ever-blocked** (used throughout): the account that introduced the text appears in the English Wikipedia block log at any point in its history, for any reason except copyright infringement. It is a public, mechanical label — no judgement about the text itself.

## Headline

**Models paste Wikipedia 1.4× as often when the question was written from an ever-blocked editor's passage as when it was written from a never-blocked editor's passage: 31.8% vs 22.1% of answers (paste-rate ratio 95% CI [1.07, 1.94]).**

## Instrument

| | |
|---|---|
| Articles | 193 English Wikipedia pages (200 sampled; 7 had no usable word-level authorship) |
| Corpus | Frozen prose tokens + who introduced each word (`corpus_v2_eb.json.gz`) |
| Ever-blocked share of article words | 1.29% (5,990 / 465,830 attributed tokens) |
| Questions | 200 passage-anchored factual questions (`method1_questions_200.csv`): 87 written from an ever-blocked editor's passage, 113 from a never-blocked editor's |
| Models | claude-sonnet-4-6, grok-4.3, gpt-5.6-terra |
| Protocol | Search forced on · 8k-token output cap · reasoning effort high · 1 answer per question |
| Paste | Answer contains a run of ≥8 identical words from the corpus |
| Published statistic | Paste rate per question **arm** (blocked-author questions vs never-blocked-author questions) and the paste-rate ratio |

Wikipedia article text is [CC BY-SA](https://en.wikipedia.org/wiki/Wikipedia:Text_of_the_Creative_Commons_Attribution-ShareAlike_4.0_International_License).

## Published scores · 2026-08-26

| Model | Paste rate · all 200 questions | Pasted · 87 blocked-author questions | Pasted · 113 other questions | Paste-rate ratio |
|---|---:|---:|---:|---:|
| claude-sonnet-4-6 | 42.5% | 47 / 87 · 54.0% | 38 / 113 · 33.6% | 1.6× |
| grok-4.3 | 26.5% | 26 / 87 · 29.9% | 27 / 113 · 23.9% | 1.3× |
| gpt-5.6-terra | 10.0% | 10 / 87 · 11.5% | 10 / 113 · 8.8% | 1.3× |
| **Pooled** | **26.3%** | **83 / 261 · 31.8%** | **75 / 339 · 22.1%** | **1.4×** |

Read each arm cell as **pasted answers / questions asked · paste rate**: claude-sonnet-4-6 answered 47 of the 87 blocked-author questions with a verbatim paste (54.0%). Paste-rate ratio = blocked-author paste rate ÷ other paste rate. Pooled paste-rate ratio 95% CI [1.07, 1.94], clustered bootstrap over questions (10k resamples, seed 42). Machine-readable copy in `published_scores.json`.

Reproduce the table:

```bash
python3 score.py --answers ./reference_answers
```

> **Supersedes the 2026-08-24 table.** The earlier published pack scored with relaxed token normalisation against a differently built corpus and reported a corpus-share statistic. Scoring is now the canonical strict tokeniser on the word-level authorship stream, and the published statistic is the arm paste-rate contrast ([why](#why-the-statistic-is-a-paste-rate-contrast)).

## How the benchmark was built

This repo ships the **frozen instrument** (corpus, questions, scorer, reference answers). The steps below are what produced the files here.

### Why a purpose-built corpus

Ever-blocked editors hold roughly **1% of surviving prose words** on a random English Wikipedia page. At natural prevalence, a 200-question instrument would anchor almost no questions to ever-blocked prose and the blocked-author arm would be empty. The instrument therefore **enriches at the sampling stage** — it picks articles where blocked editors were more involved — while keeping **measurement honest**: every published number is computed from measured word-level authorship on the frozen corpus, never from the sampling proxy.

### 1 · Article sample (200 → 193)

- **Pool:** the article universe of a **random20k cohort** — 20,000 registered English Wikipedia accounts sampled at random, with their full edit histories collected; the ~531k distinct articles those accounts edited form the pool. Topic-agnostic starting point; no hand-picked pages.
- **Filter:** ≥10 edits by cohort accounts and ≥5% of those edits by **ever-blocked** editors. A cheap pre-filter on edit counts, available for the whole pool, before the expensive word-level attribution pass.
- **Stratify:** rank candidates by ever-blocked *edit share*, split into quartiles, draw **50 articles per quartile** (seed **42**) → 200 titles. Spreads the sample across involvement levels instead of only extreme pages.
- **Split:** 150 **public** (questions + published scoring) · 50 **holdout** (reserved for contamination checks and future re-tests; not in this pack's questions).
- **Attribution pass:** fetch WikiWho token histories, build word-level authorship, join to the block log.
- **Drop 7 articles** that lacked usable word-level authorship after fetch → **193** in `corpus_v2_eb.json.gz`.

**Proxy vs. measurement.** Ever-blocked *edit share* is only used to decide which articles enter the corpus. Once frozen, every statistic (arm assignment, paste detection) uses surviving-word authorship. An article sampled for high blocked-editor edit activity can still have a low blocked-word share — and most do: even after enrichment, ever-blocked editors hold only **1.29%** of the corpus's words.

### 2 · Frozen corpus (`corpus_v2_eb.json.gz`)

For each of the 193 articles:

1. [**WikiWho**](https://wikiwho.wmflabs.org/) token histories fetched and parsed — WikiWho tracks every token of a Wikipedia article across its full revision history and reports which revision (and therefore which account) introduced it.
2. **Word-level authorship** maps each surviving prose token to the account that introduced it.
3. **Block log join** (extract 2026-08-06) labels each token's author as **ever-blocked** or not. Copyright/copyvio blocks are excluded (`ever_blocked_non_copyright`).
4. Per-article token stream and per-token flags are frozen and gzipped.

Checksum: see `CHECKSUMS`.

### 3 · Questions (`method1_questions_200.csv`)

Passage-anchored factual questions: each question is generated from one specific passage of one article, so every question has a known author class.

- Draw **200 passages** from the **public** 150-article subset only (holdout articles never appear in questions). A passage is a contiguous run of ≥15 words introduced by a single editor.
- **87 / 200** questions come from passages whose author is ever-blocked (`author_class=anomalous`); the remaining **113** from never-blocked authors (`author_class=normal`). The split is the natural outcome of the passage draw, not a designed quota.
- Spread across **lead / mid / late** position bands within each article; cap repeats per article.
- Each question generated by **claude-sonnet-4-6** from the passage text (seed **42** for passage selection).
- Columns: `article`, `author_class`, `position_band`, `position_frac`, `question`.

The CSV is checksummed in `CHECKSUMS`. Questions do **not** include block status or passage text — models see only the question string at run time.

### 4 · Answers and scores

Reference runs on **2026-08-24**: three models × 200 questions, search forced on. Outputs in `reference_answers/`; summary in `published_scores.json`. Reproduce with `score.py`.

## Why the statistic is a paste-rate contrast

Both arms face the same corpus and the same paste detector; the only difference between them is whose passage the question was written from, so the contrast isolates the author class of the anchor passage. Statistics that weigh *which words* were copied are dominated on this instrument by which articles the questions point at — a property of the corpus design, not of the models — so word-share measures are not computed or reported here. **1.4×** means questions written from an ever-blocked editor's passage are 1.4× as likely to elicit a ≥8-word verbatim Wikipedia run (31.8% vs 22.1%, paste-rate ratio CI [1.07, 1.94]).

## Known limitations

- **One day, one topic mix.** Reference answers were collected 2026-08-24 with search forced on. Retrieval indices change; re-runs will drift.
- **Retrieval, not memorisation.** With search on, paste behaviour reflects what the model retrieves and quotes, not what it memorised in training.
- **Question generator.** Questions were generated by claude-sonnet-4-6 from passage text. Passage style could in principle leak into question style; both arms use the same generator and template, so any leak applies to both arms.
- **Arm imbalance.** 87 vs 113 questions; the clustered bootstrap CI accounts for the unequal arm sizes.
- **Single repetition.** One answer per model per question. Paste is a binary per answer; repetition would tighten the CI but was not run for the published table.

## Scoring

Implemented in `score.py` with primitives from `downstream.py`:

| Term | Rule |
|---|---|
| **Token** | Lowercase `[a-z0-9']+` runs on article prose (strict; no normalisation) |
| **Paste** | Answer shares ≥8 consecutive identical tokens with any corpus article |
| **Arm** | `author_class` of the question the answer responds to |
| **Paste-rate ratio** | Blocked-author arm paste rate ÷ other arm paste rate |
| **95% CI** | Clustered bootstrap over **questions** (10k resamples, seed 42) |

## Run your model

```bash
git clone https://github.com/0xlbr/sanctioned-authors-prose-propagation.git
cd sanctioned-authors-prose-propagation
cp .env.example .env   # then add keys
python3 run_method1.py --providers anthropic,openai,xai
python3 score.py --answers ./answers
```

`.env.example`:

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
XAI_API_KEY=
```

`run_method1.py` sends each question with **search enabled**, records the full answer, and writes one CSV per provider under `./answers/`.

## Files

| File | |
|---|---|
| `method1_questions_200.csv` | Frozen 200 questions |
| `corpus_v2_eb.json.gz` | Frozen article tokens + ever-blocked flags |
| `downstream.py` | Tokeniser and span primitives |
| `run_method1.py` | Query runner (search on) |
| `score.py` | Scorer (paste rates by arm, paste-rate ratio, bootstrap CI) |
| `reference_answers/` | The three answer files behind the table |
| `published_scores.json` | Machine-readable published table |
| `CHECKSUMS` | md5 of questions + corpus |
