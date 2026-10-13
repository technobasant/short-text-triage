"""Cluster day-1 text three ways and score against the planted intents.

  (a) HDBSCAN on the raw 384-d vectors of distinct L4 keys
  (b) UMAP(10-d, cosine) -> HDBSCAN on the same distinct keys
  (c) UMAP -> HDBSCAN on every row (each row carries the vector of its raw text)

Scores are row-weighted: a cluster that is right about a string seen 900 times
matters more than one that is right about a typo seen once.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

from .embed import encode, timed
from .io import OUT, rows, save
from .normalize import key

SEED = 42
HDB = dict(min_cluster_size=5, min_samples=3)


def umap_hdbscan(x: np.ndarray) -> np.ndarray:
    import umap
    z = umap.UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric="cosine", random_state=SEED).fit_transform(x)
    return HDBSCAN(**HDB).fit_predict(z)


def hdbscan_raw(x: np.ndarray) -> np.ndarray:
    return HDBSCAN(**HDB).fit_predict(x)


def score(row_labels: np.ndarray, d1: list[dict]) -> dict:
    intents = [r["intent"] for r in d1]
    langs = [r["lang"] for r in d1]
    clustered = row_labels != -1
    by_c = defaultdict(Counter)
    lang_c = defaultdict(set)
    for c, i, l in zip(row_labels, intents, langs):
        if c != -1:
            by_c[c][i] += 1
            lang_c[c].add(l)
    purity = sum(cnt.most_common(1)[0][1] for cnt in by_c.values()) / max(1, clustered.sum())
    majority = {c: cnt.most_common(1)[0][0] for c, cnt in by_c.items()}
    per_intent = Counter(majority.values())
    idx = np.where(clustered)[0]
    return {
        "clusters": len(by_c),
        "noise_row_share": round(1 - clustered.mean(), 4),
        "purity_rows": round(purity, 4),
        "ari_clustered_rows": round(adjusted_rand_score([intents[i] for i in idx], row_labels[idx]), 4),
        "ari_all_rows_noise_as_one_bucket": round(adjusted_rand_score(intents, row_labels), 4),
        "multilingual_cluster_share": round(sum(1 for s in lang_c.values() if len(s - {"xx"}) > 1) / max(1, len(lang_c)), 4),
        "clusters_per_intent": dict(sorted(per_intent.items())),
    }


def main() -> None:
    d1 = rows("day1")
    keys = (OUT / "day1_distinct.txt").read_text(encoding="utf-8").split("\n")
    kx = np.load(OUT / "emb_day1_distinct.npy")
    kpos = {k: i for i, k in enumerate(keys)}
    row_to_key = np.array([kpos[key(r["text"])] for r in d1])

    raw_distinct = sorted({r["text"] for r in d1})
    rx = encode(raw_distinct)
    rpos = {t: i for i, t in enumerate(raw_distinct)}
    row_vecs = rx[[rpos[r["text"]] for r in d1]]

    res = {"hdbscan": HDB, "umap": {"n_neighbors": 15, "n_components": 10, "min_dist": 0.0, "metric": "cosine", "random_state": SEED},
           "load_avg_start": os.getloadavg(), "runs": 3, "results": []}

    for name, fn, x, to_rows in (
        ("(a) HDBSCAN, 384-d, distinct keys", hdbscan_raw, kx, lambda lab: lab[row_to_key]),
        ("(b) UMAP->HDBSCAN, distinct keys", umap_hdbscan, kx, lambda lab: lab[row_to_key]),
        ("(c) UMAP->HDBSCAN, every row", umap_hdbscan, row_vecs, lambda lab: lab),
    ):
        lab, wmed, walls, cmed, cpus = timed(lambda: fn(x))
        out = {"method": name, "points": len(x), "median_wall_s": round(wmed, 2), "wall_runs_s": [round(t, 2) for t in walls],
               "median_cpu_s": round(cmed, 2), "cpu_runs_s": [round(t, 2) for t in cpus]}
        out.update(score(to_rows(lab), d1))
        res["results"].append(out)
        if name.startswith("(b)"):
            np.save(OUT / "labels_day1_umap.npy", lab)

    res["load_avg_end"] = os.getloadavg()
    save("cluster", res)


if __name__ == "__main__":
    main()
