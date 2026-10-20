# short-text-triage

[![ci](https://github.com/technobasant/short-text-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/technobasant/short-text-triage/actions/workflows/ci.yml)

Short free text — support reports, reviews, a comment field nobody designed — arrives as rows, and the obvious pipeline sends each row to a model. That pays for the same meaning thousands of times, labels it inconsistently, and, once you switch to embeddings to save money, files anything new under the nearest category it already knows with a score that looks confident.

This lab tests the alternative on a generated multilingual corpus. **Normalize to a key, embed each distinct key once, cluster the keys, and label the clusters. Then assign a new string only when its best label beats the runner-up by a margin.** Strings that fail the margin go to a review pool that is itself clustered. A model never sees a row.

Write-up: [Label the string, not the row: a short-text triage lab](https://basantbhattarai.com.np/writing/classify-distinct-strings-not-rows/)

## What is in the corpus

Everything is **generated and reproducible**; no real user text is used anywhere. [`triage/phrases.py`](triage/phrases.py) holds 240 hand-written base phrasings — 8 support-inbox intents × 5 languages (en, es, de, fr, pt) × 6 wordings — plus a ninth intent that only appears on day 2. [`triage/corpus.py`](triage/corpus.py) damages them the way free-text fields get damaged (case, stray and non-breaking spaces, trailing punctuation, NFD-decomposed accents, single-character typos) and samples them Zipf-style into two 20,000-row days.

Day 2 is the test: 14,086 rows of known wordings with new damage, 3,937 rows of wordings held back from day 1, 1,363 rows of an intent that has no cluster at all, and 614 rows of noise.

## Results

Every figure is from [MEASUREMENTS.md](MEASUREMENTS.md), which cites the JSON in [`out/results/`](out/results) it came from.

| Question | Measured |
|---|---|
| How much does normalization shrink the work? | 3,018 raw distinct strings → **1,537** keys after whitespace, NFKC, casefold and edge punctuation (49% fewer); 88.7% of the remaining keys are typos |
| What does skipping NFKC cost? | A source that switches to NFD: a raw-key cache misses **613 of 613** of its rows; the normalized key misses 0 |
| Embedding cost, keys vs rows | **6.74 vs 93.31** CPU seconds (13.8×) |
| Clustering keys vs rows (UMAP → HDBSCAN) | 124 clusters vs **1,254** — 1,254 labelling calls; 3.30 vs 35.44 CPU s; multilingual clusters 25.8% vs 1.0% |
| How tight is the cosine band? | **76.4%** of different-intent pairs score ≥ 0.80; same-intent median 0.854 vs different-intent 0.819 |
| Nearest-centroid accuracy on day-2 misses | typos of known wordings 97.2%; unseen wordings **55.0%**; the new intent 0% (it has no cluster) |
| Best absolute threshold (τ = 0.92) | 66.2% precision on 2,862 assigned rows |
| Margin rule (m = 0.03) | **95.2%** precision on 1,728 assigned rows; 48 of 1,363 new-intent rows leaked |
| Does the new intent surface on its own? | Review pool: 515 keys → 44 clusters; 14 of them hold 1,298 of the 1,315 new-intent rows that reached the pool, at 97.2% purity |
| LLM calls, day 1 | per row 20,000 · per raw string 3,018 · per key 1,537 · **per cluster 125** |

Two expected results did **not** reproduce, and the lab says so rather than burying them:

- **UMAP added no purity at this size.** HDBSCAN on the raw 384-dimensional vectors reached 99.56% purity against 99.45% after UMAP; UMAP only cut the noise share from 4.87% to 2.12% and produced more multilingual clusters.
- **Dropping e5's `query: ` prefix did not hurt nearest-neighbour precision** (P@1 0.963 without it, 0.933 with it). What the prefix does do is move the same text by 0.017–0.057 in cosine — close to the 0.03 decision margin — so the rule that survives is *keep the prefix consistent*, not *always use it*.

One caveat shapes the threshold story. A single 830-row French login-failure wording sits at cosine 0.953 to the account-takeover cluster with a margin of 0.0298. It drives both the 63.4% → 95.2% step between m = 0.02 and m = 0.03, and the precision *drop* to 54.2% at τ = 0.94.

## Reproduce

```bash
uv sync --python 3.12
./run_all.sh     # corpus → dedupe → bench_embed → cluster → prefix → assign → calls
```

Each step is also a module (`uv run python -m triage.<step>`), and later steps read what earlier ones wrote to `out/`. The model is [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small), pinned to a revision in [`triage/embed.py`](triage/embed.py), run on CPU; the first run downloads it.

**CI reproduces the deterministic half on every push.** The corpus and the dedupe step are seeded and use only the standard library, so CI regenerates them and fails unless section 1 of MEASUREMENTS.md comes back exactly. The model steps are not rerun in CI — they need torch and the model download, take minutes of CPU, and their timings are hardware-specific — so run `./run_all.sh` locally for those.

## Layout

| Path | What it does |
|---|---|
| `triage/phrases.py` | base phrasings; which wordings are day-1, held back, or new |
| `triage/corpus.py` | seeded damage + Zipf sampling → `out/day1.csv`, `out/day2.csv` |
| `triage/normalize.py` | cache-key levels L0–L4 |
| `triage/dedupe.py` | distinct strings per level; day-2 cache hit rates; NFD misses |
| `triage/bench_embed.py` | embedding cost: distinct keys vs every row |
| `triage/cluster.py` | HDBSCAN on 384-d vs UMAP → HDBSCAN; distinct keys vs rows |
| `triage/prefix.py` | the e5 `query: ` prefix effect; the cosine band |
| `triage/assign.py` | day-2 assignment: absolute threshold vs margin bands |
| `triage/calls.py` | LLM-call arithmetic; whether the new intent surfaces on its own |
| `out/results/*.json` | the raw measurements MEASUREMENTS.md cites |
| `out/*.log` | the logs of the measured runs, with the load average per step |

## Method notes

- **Apparatus:** Apple M5 (4 performance + 6 efficiency cores), Python 3.12 via `uv`, torch 2.14, sentence-transformers 6.1, scikit-learn 1.9 (`sklearn.cluster.HDBSCAN`), umap-learn 0.5.12; 4 torch threads.
- **The machine was shared.** The 1-minute load average was 34–56 during the timed runs, so wall-clock figures are inflated. **Process CPU time is the comparable figure**, and every timing is the median of 3 runs.
- **Clustering:** HDBSCAN `min_cluster_size=5, min_samples=3`; UMAP `n_neighbors=15, n_components=10, min_dist=0.0, metric=cosine, random_state=42`. Purity is row-weighted against the planted intents. Low ARI with high purity is intended: each intent splits into 9–21 fine, wording-level clusters, which become subcategories.
- **Assignment:** each day-1 cluster is labelled with its row-weighted majority intent (standing in for one LLM call per cluster plus human approval). For a miss, s1 is the top-1 centroid cosine and s2 the best centroid with a *different* label; the margin is s1 − s2.

## Limits

- **Generated corpus.** Hand-written phrasings and mechanical damage are cleaner than real inboxes; what transfers is the mechanism (normalization collapses variants, the cosine band is narrow, a margin beats a threshold), not the exact percentages.
- **One model, one machine.** multilingual-e5-small on CPU on one shared machine (Apple M5), under load. Larger or domain-tuned embedders will move the band.
- **Small n where it matters.** The threshold story turns on one heavy key; the lab reports that rather than averaging it away.
- **Built from scratch.** The question comes from [the consulting engagement](https://basantbhattarai.com.np/work/consulting-short-text-and-alerting/) this lab rebuilds the idea of; no data, code or domain from it is used here.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
