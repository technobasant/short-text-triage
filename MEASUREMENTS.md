# Measurements

Every figure the article cites, with the method behind it. Raw outputs are in
`out/results/*.json`; regenerate with `./run_all.sh`. **The corpus is
generated** (`triage/phrases.py`, `triage/corpus.py`); no real user text was
used anywhere.

## Apparatus

| | |
|---|---|
| Machine | Apple M5, 10 CPU cores (4 performance + 6 efficiency), 24 GB, macOS (Darwin 27) |
| Python | 3.12.14 via `uv` |
| Packages | torch 2.14.0 · sentence-transformers 6.1.0 · scikit-learn 1.9.1 (`sklearn.cluster.HDBSCAN`) · umap-learn 0.5.12 · numpy 2.5.3 |
| Model | `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, CPU, fp32, 384-d, L2-normalised |
| Threads | `torch.set_num_threads(4)`, `TOKENIZERS_PARALLELISM=false` |
| Load | **The machine was shared.** 1-minute load average 34–56 during the timed runs (recorded per run in the JSON). Wall-clock figures are therefore inflated and noisy; **process CPU time** is reported alongside and is the figure to compare. |
| Timings | every timing is the median of 3 runs |

## Corpus (`triage/corpus.py`, seeds 101 / 202, popularity seed 7)

- 8 intents × 5 languages (en, es, de, fr, pt) × 6 hand-written wordings = 240 base phrasings, plus 14 content-free tokens labelled `noise`.
- Day 1: 20,000 rows; wordings 0–3 of each intent/language only (160 wordings) at 97%, noise 3%.
- Day 2: 20,000 rows; 14,086 day-1 wordings, 3,937 held-back wordings (4–5), 1,363 rows of a ninth intent (`change_address`) that never appears on day 1, 614 noise.
- Damage mix on day 1: exact 9,633 · case 3,039 · typo 2,426 · trailing punctuation 2,384 · whitespace/NBSP 1,640 · NFD 878.
- Wordings drawn Zipf (a = 1.1) so a few dominate. Language mix on day 1 follows from that draw: pt 5,853 · fr 5,784 · de 3,346 · en 2,402 · es 2,009 · noise 606.

## 1. Rows → distinct strings (`dedupe.json`)

| Key | Day-1 distinct | Rows per key | Day-2 row cache hit rate |
|---|---:|---:|---:|
| L0 raw | 3,018 | 6.63 | 67.76% |
| L1 whitespace collapsed | 2,624 | 7.62 | 68.45% |
| L2 + NFKC | 2,569 | 7.79 | 68.49% |
| L3 + casefold | 2,182 | 9.17 | 68.96% |
| L4 + edge punctuation | 1,537 | 13.01 | 70.07% |

- L4 leaves 49% fewer keys than raw (1,537 vs 3,018).
- Composition of the 1,537 L4 keys by the variant that produced their first occurrence: typo 1,364 (88.7%), exact 98, punct 28, case 26, space 17, NFD 4. The 173 non-typo keys are the 160 day-1 wordings plus 13 noise keys — L4 collapses every mechanical variant back to its base wording; what stays distinct is typos.
- `?` and `...` both normalise to the empty string. The pipeline must drop empty keys before they get a label.
- `casefold()` vs `lower()`: `"Straße".casefold()` → `strasse`; `.lower()` → `straße`.
- Day-2 hit-rate ceiling: day-1 wordings are 70.4% of day-2 rows, so 70.07% at L4 is close to it. In steady state, a raw cache has *also* seen most variants, which is why the raw-vs-L4 hit-rate gap is only 2.3 points here.

### NFD (`dedupe.json` → `nfd`, `nfd_shift_scenario`)

- Steady state: of 613 day-2 NFD rows whose wording appears on day 1, a raw-key cache misses 8 (1.3%); the L4 cache misses 0. Day 1 had already seen NFD renderings of the popular wordings.
- Shift scenario (day-1 cache filled only from sources that sent NFC; one source starts sending NFD on day 2): raw-key cache misses 613 of 613 (100%); L4 misses 0.

## 2. Embedding cost (`embed_throughput.json`, batch 64, 4 threads)

| Input | Texts | Median CPU s | Median wall s | CPU ms / text |
|---|---:|---:|---:|---:|
| Day-1 distinct L4 keys | 1,537 | 6.74 | 28.53 | 4.38 |
| Day-1 rows | 20,000 | 93.31 | 387.18 | 4.67 |

CPU-time ratio rows / keys = 13.8×. Wall runs: keys 40.96 / 28.53 / 28.32 s; rows 346.09 / 387.18 / 431.29 s (load 37–50).

## 3. Clustering day 1 (`cluster.json`)

HDBSCAN `min_cluster_size=5, min_samples=3`; UMAP `n_neighbors=15, n_components=10, min_dist=0.0, metric=cosine, random_state=42`. Scores are **row-weighted** against the planted intents. (c) gives each row the vector of its raw text.

| Method | Points | Median CPU s | Clusters | Noise rows | Purity | ARI (clustered rows) | Multilingual clusters |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) HDBSCAN on 384-d, distinct keys | 1,537 | 1.22 | 97 | 4.87% | 99.56% | 0.574 | 19.6% |
| (b) UMAP → HDBSCAN, distinct keys | 1,537 | 3.30 | 124 | 2.12% | 99.45% | 0.487 | 25.8% |
| (c) UMAP → HDBSCAN, every row | 20,000 | 35.44 | 1,254 | 11.91% | 99.98% | 0.011 | 1.0% |

- ARI is low for (a) and (b) because each intent splits into 9–21 fine clusters (wording-level); purity says those clusters are clean. Fine clusters are the intended output — they become subcategories.
- (c): exact duplicates satisfy `min_cluster_size` on their own, so popular raw strings become their own clusters; 1,254 clusters means 1,254 labelling calls.
- On this corpus UMAP did **not** improve purity over plain HDBSCAN at 384-d; it cut noise from 4.87% to 2.12% and produced more multilingual clusters.

## 4. The e5 prefix (`prefix_cosine.json`)

240 clean base wordings; nearest neighbour excludes the probe itself.

| Prefix | P@1 | Cross-lingual P@1 |
|---|---:|---:|
| `query: ` | 0.933 | 0.975 |
| `passage: ` | 0.896 | 0.992 |
| none | 0.963 | 0.971 |

Mixed prefixes (index embedded one way, probe another):

| Index | Probe | P@1 | Mean cosine of the same text across the two prefixes |
|---|---|---:|---:|
| `query: ` | `query: ` | 0.933 | 1.000 |
| `query: ` | none | 0.946 | 0.983 |
| `query: ` | `passage: ` | 0.921 | 0.943 |
| none | `query: ` | 0.938 | 0.983 |

No prefix penalty reproduced on P@1. The same text moves by 0.017 (query↔none) to 0.057 (query↔passage) in cosine when the prefix changes.

## 5. The cosine band (`prefix_cosine.json`, `query: ` prefix, 28,680 pairs)

| Pairs | p1 | p5 | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| all | 0.761 | 0.777 | 0.822 | 0.879 | 0.915 |
| same intent | 0.781 | 0.802 | 0.854 | 0.926 | 0.953 |
| different intent | 0.760 | 0.776 | 0.819 | 0.866 | 0.887 |
| short (≤ 3 words) same intent | 0.789 | 0.813 | 0.873 | 0.947 | 0.964 |
| short (≤ 3 words) different intent | 0.766 | 0.779 | 0.819 | 0.871 | 0.888 |

- 76.4% of different-intent pairs score ≥ 0.80; 44.8% of same-intent pairs score < 0.85. Min over all pairs 0.715.
- Histogram, 0.05 bins from 0.70: same intent 2 / 152 / 1,405 / 1,441 / 433 / 47; different intent 68 / 5,880 / 15,755 / 3,413 / 83 / 1.

## 6. Day-2 assignment (`assign.json`)

Reference: the 124 day-1 clusters from (b), each labelled with its row-weighted majority intent (stand-in for one LLM call per cluster plus human approval); centroid = normalised mean of member vectors. The label cache holds day-1 keys that landed in a labelled cluster.

- Cache hits: 574 keys, 13,718 rows, label precision 99.52%.
- Cache misses: 1,243 keys, 6,282 rows → embedded and scored against centroids. s1 = top-1 cosine; s2 = best centroid with a *different* label; margin = s1 − s2.
- Nearest-centroid accuracy on misses, by row source: typos of day-1 wordings 875 / 900 (97.2%); held-back wordings 2,167 / 3,937 (55.0%); novel intent 0 / 1,363 (no cluster exists); noise 82 / 82.
- s1 percentiles, all misses: p5 0.873 · p50 0.938 · p95 0.991. Novel-intent misses: p5 0.861 · p50 0.890 · p95 0.911.
- Margin percentiles: correct top-1 p5 0.007 · p50 0.065 · p95 0.108; wrong top-1 p5 0.001 · p50 0.010 · p95 0.033.
- Heaviest wrong key: `je n'arrive pas à me connecter à mon compte` (login failure), 830 rows, top-1 `account_takeover` at s1 = 0.9533, s2 = 0.9235, margin = 0.0298.

Absolute threshold (assign if s1 ≥ τ, else outlier):

| τ | Assigned rows | Precision | Novel rows assigned | Outlier rows |
|---:|---:|---:|---:|---:|
| 0.80 | 6,282 | 49.7% | 1,363 | 0 |
| 0.85 | 6,253 | 50.0% | 1,358 | 29 |
| 0.88 | 5,908 | 49.8% | 1,219 | 374 |
| 0.90 | 4,424 | 57.8% | 165 | 1,858 |
| 0.92 | 2,862 | 66.2% | 0 | 3,420 |
| 0.94 | 1,982 | 54.2% | 0 | 4,300 |

Margin bands (outlier if s1 < 0.80; assigned if margin ≥ m; else review):

| m | Assigned rows | Precision | Novel rows assigned | Review rows |
|---:|---:|---:|---:|---:|
| 0.00 | 6,282 | 49.7% | 1,363 | 0 |
| 0.01 | 4,770 | 60.1% | 498 | 1,512 |
| 0.02 | 2,994 | 63.4% | 162 | 3,288 |
| 0.03 | 1,728 | 95.2% | 48 | 4,554 |
| 0.04 | 1,354 | 95.9% | 46 | 4,928 |
| 0.05 | 1,210 | 95.6% | 46 | 5,072 |
| 0.08 | 399 | 100.0% | 0 | 5,883 |

- The 63.4% → 95.2% step between m = 0.02 and 0.03 is mostly the single 830-row key above (margin 0.0298) crossing into review.
- At τ = 0.94 precision *falls* to 54.2% because that same key (s1 0.953) is above the threshold and makes up 830 of the 1,982 assigned rows.
- No outliers at floor 0.80: every miss scored s1 ≥ 0.80, so on this corpus the floor caught nothing. It is still the right place for content-free input, but the margin did all the work.

## 7. Discovery and LLM-call arithmetic (`calls.json`)

- Review + outlier pool at (floor 0.80, m 0.03): 515 keys → UMAP → HDBSCAN → 44 clusters (7.4% of keys left as noise).
- 14 of the 44 clusters are majority `change_address`; they hold 1,298 of the 1,315 novel rows that reached the pool, at 97.2% row-weighted purity.

| Unit of work | Day 1 calls | Day 2 calls |
|---|---:|---:|
| per row | 20,000 | 20,000 |
| per distinct raw string | 3,018 | — |
| per distinct L4 key | 1,537 | 1,243 (cache misses) |
| per cluster (+ 1 taxonomy call on day 1) | 125 | 44 (discovery clusters; assigned rows cost 0) |
