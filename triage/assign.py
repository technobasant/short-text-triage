"""Day 2: place strings the cache has never seen, two ways.

  absolute  assign to the nearest cluster if top-1 cosine >= tau, else outlier
  margin    outlier if top-1 < floor; assign only if (top-1 - best other label) >= m; else review

Reference clusters are the day-1 UMAP->HDBSCAN clusters, each labelled with the
majority planted intent — standing in for "one LLM call per cluster, then a
human approves the label". Day 2 carries wordings held back from day 1 and a
ninth intent (change_address) that has no cluster at all: every one of those
rows that gets *assigned* is a confident wrong answer.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .embed import encode
from .io import OUT, rows, save
from .normalize import key


def reference():
    keys = (OUT / "day1_distinct.txt").read_text(encoding="utf-8").split("\n")
    kx = np.load(OUT / "emb_day1_distinct.npy")
    lab = np.load(OUT / "labels_day1_umap.npy")
    d1 = rows("day1")
    votes = defaultdict(Counter)
    kpos = {k: i for i, k in enumerate(keys)}
    for r in d1:
        c = lab[kpos[key(r["text"])]]
        if c != -1:
            votes[c][r["intent"]] += 1
    clabel = {c: v.most_common(1)[0][0] for c, v in votes.items()}
    cids = sorted(clabel)
    cent = np.stack([kx[lab == c].mean(0) for c in cids])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    cache = {k: clabel[lab[i]] for i, k in enumerate(keys) if lab[i] != -1}
    return cache, cent, [clabel[c] for c in cids]


def main() -> None:
    cache, cent, clabels = reference()
    d2 = rows("day2")
    weight = Counter(key(r["text"]) for r in d2)
    truth = {}
    for r in d2:
        truth.setdefault(key(r["text"]), Counter())[r["intent"]] += 1
    truth = {k: c.most_common(1)[0][0] for k, c in truth.items()}

    hit_keys = [k for k in weight if k in cache]
    miss = sorted(k for k in weight if k not in cache)
    x = encode(miss)
    s = x @ cent.T
    labels = np.array(clabels)
    top = s.argmax(1)
    s1 = s[np.arange(len(miss)), top]
    l1 = labels[top]
    s2 = np.where(labels[None, :] == l1[:, None], -9, s).max(1)
    margin = s1 - s2
    w = np.array([weight[k] for k in miss])
    t = np.array([truth[k] for k in miss])
    src = {}
    for r in d2:
        src.setdefault(key(r["text"]), Counter())[r["source"]] += 1
    src = np.array([src[k].most_common(1)[0][0] for k in miss])
    novel = t == "change_address"
    correct = l1 == t

    def summarize(assigned, review, outlier):
        a_rows = int(w[assigned].sum())
        return {
            "assigned_rows": a_rows,
            "assigned_precision_rows": round(float(w[assigned & correct].sum() / a_rows), 4) if a_rows else None,
            "novel_rows_assigned": int(w[assigned & novel].sum()),
            "novel_rows_total": int(w[novel].sum()),
            "review_rows": int(w[review].sum()),
            "outlier_rows": int(w[outlier].sum()),
            "known_intent_rows_assigned_correctly": int(w[assigned & correct].sum()),
            "known_intent_rows_total": int(w[~novel & (t != "noise")].sum()),
            # where the wrong answers come from: typos of day-1 wordings, or
            # wordings the cache has never seen
            "held_rows_assigned": int(w[assigned & (src == "held")].sum()),
            "held_rows_assigned_correctly": int(w[assigned & correct & (src == "held")].sum()),
            "held_rows_total": int(w[src == "held"].sum()),
            "known_typo_rows_assigned": int(w[assigned & (src == "known")].sum()),
            "known_typo_rows_assigned_correctly": int(w[assigned & correct & (src == "known")].sum()),
            "known_typo_rows_total": int(w[src == "known"].sum()),
        }

    out = {"cache_hit_keys": len(hit_keys), "cache_hit_rows": int(sum(weight[k] for k in hit_keys)),
           "cache_hit_precision_rows": round(sum(weight[k] for k in hit_keys if cache[k] == truth[k]) / max(1, sum(weight[k] for k in hit_keys)), 4),
           "miss_keys": len(miss), "miss_rows": int(w.sum()),
           "top1_percentiles": {f"p{q}": round(float(np.percentile(s1, q)), 3) for q in (5, 25, 50, 75, 95)},
           "top1_novel_percentiles": {f"p{q}": round(float(np.percentile(s1[novel], q)), 3) for q in (5, 25, 50, 75, 95)},
           "margin_percentiles_correct": {f"p{q}": round(float(np.percentile(margin[correct], q)), 3) for q in (5, 25, 50, 75, 95)},
           "margin_percentiles_wrong": {f"p{q}": round(float(np.percentile(margin[~correct], q)), 3) for q in (5, 25, 50, 75, 95)},
           "nearest_centroid_accuracy_by_source": {
               sname: {"rows": int(w[src == sname].sum()), "correct": int(w[(src == sname) & correct].sum())}
               for sname in ("known", "held", "novel", "noise")},
           "absolute": [], "margin": []}
    # The single most expensive wrong answer: the heaviest miss key with a wrong top-1.
    wrong = np.where(~correct & ~novel)[0]
    worst = wrong[np.argmax(w[wrong])]
    out["heaviest_wrong_key"] = {"text": miss[worst], "rows": int(w[worst]), "truth": str(t[worst]), "top1_label": str(l1[worst]),
                                 "s1": round(float(s1[worst]), 3), "s2": round(float(s2[worst]), 3), "margin": round(float(margin[worst]), 3)}
    none = np.zeros(len(miss), bool)
    for tau in (0.80, 0.85, 0.88, 0.90, 0.92, 0.94):
        a = s1 >= tau
        out["absolute"].append({"tau": tau} | summarize(a, none, ~a))
    floor = 0.80
    for m in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08):
        o = s1 < floor
        a = ~o & (margin >= m)
        out["margin"].append({"floor": floor, "m": m} | summarize(a, ~o & ~a, o))
    np.save(OUT / "day2_miss_vecs.npy", x)
    (OUT / "day2_miss.txt").write_text("\n".join(miss), encoding="utf-8")
    np.save(OUT / "day2_miss_s1.npy", s1)
    np.save(OUT / "day2_miss_margin.npy", margin)
    save("assign", out)


if __name__ == "__main__":
    main()
