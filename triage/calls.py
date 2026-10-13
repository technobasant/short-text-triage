"""LLM-call arithmetic, and whether the novel intent surfaces on its own.

No LLM is called anywhere in this lab. These are the calls you *would* make
under each unit of work, counted from the same corpus.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import HDBSCAN

from .io import OUT, rows, save
from .normalize import key


def main() -> None:
    d1, d2 = rows("day1"), rows("day2")
    lab = np.load(OUT / "labels_day1_umap.npy")
    clusters = len(set(lab.tolist()) - {-1})
    keys1 = {key(r["text"]) for r in d1}

    # Day-2 discovery: cluster the strings the margin policy refused to assign
    # (review + outlier) and count how cleanly the unseen intent comes out.
    miss = (OUT / "day2_miss.txt").read_text(encoding="utf-8").split("\n")
    x = np.load(OUT / "day2_miss_vecs.npy")
    s1 = np.load(OUT / "day2_miss_s1.npy")
    margin = np.load(OUT / "day2_miss_margin.npy")
    floor, m = 0.80, 0.03
    pool = (s1 < floor) | (margin < m)
    truth = {}
    for r in d2:
        truth.setdefault(key(r["text"]), Counter())[r["intent"]] += 1
    truth = {k: c.most_common(1)[0][0] for k, c in truth.items()}
    weight = Counter(key(r["text"]) for r in d2)
    import umap
    px = x[pool]
    z = umap.UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric="cosine", random_state=42).fit_transform(px)
    plab = HDBSCAN(min_cluster_size=5, min_samples=3).fit_predict(z)
    pkeys = [k for k, p in zip(miss, pool) if p]
    comp = {}
    for c in sorted(set(plab.tolist()) - {-1}):
        cnt = Counter()
        for k, l in zip(pkeys, plab):
            if l == c:
                cnt[truth[k]] += weight[k]
        comp[int(c)] = dict(cnt.most_common())
    novel_clusters = [c for c, v in comp.items() if max(v, key=v.get) == "change_address"]
    novel_rows_pool = sum(weight[k] for k in pkeys if truth[k] == "change_address")
    novel_rows_in_novel_clusters = sum(comp[c].get("change_address", 0) for c in novel_clusters)
    novel_cluster_purity = (novel_rows_in_novel_clusters / sum(sum(comp[c].values()) for c in novel_clusters)) if novel_clusters else None

    out = {
        "day1": {"per_row": len(d1), "per_distinct_raw": len({r['text'] for r in d1}), "per_distinct_l4": len(keys1),
                 "per_cluster_plus_taxonomy": clusters + 1},
        "day2": {"per_row": len(d2), "per_cache_miss_l4_key": len(miss),
                 "discovery_pool_keys": int(pool.sum()), "discovery_clusters": len(comp),
                 "per_discovery_cluster": len(comp)},
        "discovery": {"policy": {"floor": floor, "m": m}, "clusters": comp, "novel_majority_clusters": novel_clusters,
                      "novel_rows_in_pool": int(novel_rows_pool), "novel_rows_in_novel_majority_clusters": int(novel_rows_in_novel_clusters),
                      "novel_majority_cluster_purity_rows": round(novel_cluster_purity, 4) if novel_cluster_purity else None,
                      "pool_noise_share_keys": round(float((plab == -1).mean()), 4)},
    }
    save("calls", out)


if __name__ == "__main__":
    main()
