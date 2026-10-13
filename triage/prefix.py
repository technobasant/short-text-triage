"""What the e5 instruction prefix does, and how narrow the cosine band is.

Uses the 240 clean base phrasings (8 intents x 5 languages x 6) so a
nearest neighbour is always a *different wording*, never a typo of itself.
"""
from __future__ import annotations

import numpy as np

from .embed import encode
from .io import save
from .phrases import KNOWN, LANGS


def base() -> list[tuple[str, str, str]]:
    return [(intent, lang, p) for intent, by in KNOWN.items() for lang in LANGS for p in by[lang]]


def p_at_1(x: np.ndarray, intents: list[str], langs: list[str], cross: bool) -> float:
    s = x @ x.T
    np.fill_diagonal(s, -9)
    if cross:
        same_lang = np.equal.outer(np.array(langs), np.array(langs))
        s = np.where(same_lang, -9, s)
    nn = s.argmax(1)
    return float(np.mean([intents[i] == intents[j] for i, j in enumerate(nn)]))


def pct(v: np.ndarray) -> dict:
    return {f"p{q}": round(float(np.percentile(v, q)), 3) for q in (1, 5, 25, 50, 75, 95, 99)} | {
        "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)}


def main() -> None:
    b = base()
    intents, langs, texts = zip(*b)
    out = {"phrasings": len(b), "prefix": []}
    vecs = {}
    for prefix in ("query: ", "passage: ", ""):
        x = encode(list(texts), prefix=prefix)
        vecs[prefix] = x
        out["prefix"].append({"prefix": repr(prefix), "nn_p_at_1": round(p_at_1(x, intents, langs, False), 4),
                              "cross_lingual_p_at_1": round(p_at_1(x, intents, langs, True), 4)})

    # The failure that is actually silent: index embedded with one prefix,
    # probes embedded with another. Nearest neighbour excludes the probe's own
    # phrasing, so this is "does the new wording find its intent".
    out["mixed_prefix"] = []
    for idx_p, probe_p in (("query: ", "query: "), ("query: ", ""), ("query: ", "passage: "), ("", "query: ")):
        a, b = vecs[idx_p], vecs[probe_p]
        sm = b @ a.T
        np.fill_diagonal(sm, -9)
        nn = sm.argmax(1)
        self_cos = float(np.mean(np.sum(a * b, axis=1)))
        out["mixed_prefix"].append({"index_prefix": repr(idx_p), "probe_prefix": repr(probe_p),
                                    "p_at_1": round(float(np.mean([intents[i] == intents[j] for i, j in enumerate(nn)])), 4),
                                    "mean_cos_same_text_across_prefixes": round(self_cos, 4)})

    x = vecs["query: "]
    s = x @ x.T
    iu = np.triu_indices(len(b), 1)
    same = np.equal.outer(np.array(intents), np.array(intents))[iu]
    short = np.array([len(t.split()) <= 3 for t in texts])
    both_short = np.logical_and.outer(short, short)[iu]
    sv = s[iu]
    out["cosine_query_prefix"] = {
        "all_pairs": pct(sv),
        "same_intent": pct(sv[same]),
        "different_intent": pct(sv[~same]),
        "short_pairs_same_intent": pct(sv[same & both_short]),
        "short_pairs_different_intent": pct(sv[~same & both_short]),
        "share_of_different_intent_pairs_above_0.80": round(float((sv[~same] >= 0.80).mean()), 4),
        "share_of_same_intent_pairs_below_0.85": round(float((sv[same] < 0.85).mean()), 4),
    }
    edges = np.round(np.arange(0.70, 1.0001, 0.05), 2)
    out["histogram_query_prefix"] = {
        "bins": [f"{a:.2f}-{b:.2f}" for a, b in zip(edges[:-1], edges[1:])],
        "same_intent": np.histogram(sv[same], bins=edges)[0].tolist(),
        "different_intent": np.histogram(sv[~same], bins=edges)[0].tolist(),
        "below_0.70_same": int((sv[same] < 0.70).sum()), "below_0.70_different": int((sv[~same] < 0.70).sum()),
    }
    save("prefix_cosine", out)


if __name__ == "__main__":
    main()
