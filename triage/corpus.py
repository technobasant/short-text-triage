"""Seeded, deterministic SYNTHETIC corpus generator.

Each row is a base phrasing passed through the kind of damage real free-text
fields collect: case changes, stray and non-breaking spaces, trailing
punctuation, decomposed Unicode (NFD, as some keyboards and copy-paste paths
emit), and single-character typos. Base phrasings are drawn Zipf-style so a few
wordings dominate, as they do in real inboxes.

    uv run python -m triage.corpus          # writes out/day1.csv, out/day2.csv
"""
from __future__ import annotations

import csv
import random
import unicodedata
from pathlib import Path

from .phrases import HELD_IDX, KNOWN, KNOWN_IDX, LANGS, NOISE, NOVEL

OUT = Path(__file__).resolve().parents[1] / "out"

# Variant mix. NFD only bites when the string has a precomposed character; for
# pure-ASCII strings that draw falls through to the exact form.
VARIANTS = [("exact", 0.45), ("case", 0.15), ("punct", 0.12), ("space", 0.08), ("nfd", 0.08), ("typo", 0.12)]


def _typo(s: str, rng: random.Random) -> str:
    idx = [i for i, c in enumerate(s) if c.isalpha()]
    if len(idx) < 4:
        return s
    i = rng.choice(idx[1:-1])
    op = rng.choice(["swap", "drop", "dup"])
    if op == "swap" and i + 1 < len(s):
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op == "drop":
        return s[:i] + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def damage(s: str, rng: random.Random) -> tuple[str, str]:
    r, acc = rng.random(), 0.0
    for kind, p in VARIANTS:
        acc += p
        if r < acc:
            break
    if kind == "case":
        s = rng.choice([s.upper(), s.lower(), s[:1].upper() + s[1:], s.title()])
    elif kind == "punct":
        s = s + rng.choice(["!", "!!", "!!!", "...", "?", " ?", "."])
    elif kind == "space":
        s = rng.choice([" " + s, s + "  ", s.replace(" ", "  ", 1), s.replace(" ", " ", 1)])
    elif kind == "nfd":
        nfd = unicodedata.normalize("NFD", s)
        kind = "nfd" if nfd != s else "exact"
        s = nfd
    elif kind == "typo":
        s = _typo(s, rng)
    return s, kind


def _zipf_weights(n: int, a: float = 1.1) -> list[float]:
    return [1.0 / (k ** a) for k in range(1, n + 1)]


def _pool(intents: dict, idx: list[int], rng: random.Random) -> list[tuple[str, str, str, int]]:
    """(intent, lang, phrase, phrase_index) — shuffled so Zipf rank is not phrase order."""
    pool = [(intent, lang, by_lang[lang][i], i) for intent, by_lang in intents.items() for lang in LANGS for i in idx]
    rng.shuffle(pool)
    return pool


def generate(n: int, seed: int, mix: dict[str, float]) -> list[dict]:
    rng = random.Random(seed)
    # Popularity is fixed by a separate seed so day 1 and day 2 agree on which
    # wordings are common, as two days of the same inbox would.
    pop = random.Random(7)
    known = _pool(KNOWN, KNOWN_IDX, pop)
    held = _pool(KNOWN, HELD_IDX, pop)
    novel = _pool(NOVEL, KNOWN_IDX + HELD_IDX, pop)
    pools = {"known": (known, _zipf_weights(len(known))),
             "held": (held, _zipf_weights(len(held))),
             "novel": (novel, _zipf_weights(len(novel)))}
    rows = []
    for _ in range(n):
        r, acc = rng.random(), 0.0
        for src, p in mix.items():
            acc += p
            if r < acc:
                break
        if src == "noise":
            text, kind = damage(rng.choice(NOISE), rng)
            rows.append({"text": text, "intent": "noise", "lang": "xx", "source": "noise", "base": "", "variant": kind})
            continue
        pool, w = pools[src]
        intent, lang, base, _ = rng.choices(pool, weights=w, k=1)[0]
        text, kind = damage(base, rng)
        rows.append({"text": text, "intent": intent, "lang": lang, "source": src, "base": base, "variant": kind})
    return rows


DAY1_MIX = {"known": 0.97, "noise": 0.03}
DAY2_MIX = {"known": 0.70, "held": 0.20, "novel": 0.07, "noise": 0.03}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, seed, mix in (("day1", 101, DAY1_MIX), ("day2", 202, DAY2_MIX)):
        rows = generate(20_000, seed, mix)
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
