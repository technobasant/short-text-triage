"""How many distinct strings each normalization level leaves, and what it does
to a label cache built on day 1 and read on day 2."""
from __future__ import annotations

from collections import Counter

from .io import rows, save
from .normalize import LEVELS


def main() -> None:
    d1, d2 = rows("day1"), rows("day2")
    out = {"rows_day1": len(d1), "rows_day2": len(d2), "levels": []}
    for name, f in LEVELS.items():
        k1 = {f(r["text"]) for r in d1}
        k2 = [f(r["text"]) for r in d2]
        hits = sum(1 for k in k2 if k in k1)
        d2_distinct = set(k2)
        out["levels"].append({
            "level": name,
            "distinct_day1": len(k1),
            "rows_per_distinct_day1": round(len(d1) / len(k1), 2),
            "day2_row_cache_hit_rate": round(hits / len(k2), 4),
            "day2_distinct": len(d2_distinct),
            "day2_distinct_misses": len(d2_distinct - k1),
        })

    # The silent-miss case: day-2 rows that are an NFD rendering of a string
    # whose NFC form the cache already holds.
    raw1 = {r["text"] for r in d1}
    # Only rows whose base wording was seen on day 1: a miss here is the
    # normalization's fault, not a genuinely new wording.
    nfd_rows = [r for r in d2 if r["variant"] == "nfd" and r["source"] == "known"]
    nfd_raw_miss = sum(1 for r in nfd_rows if r["text"] not in raw1)
    l4 = LEVELS["L4 +edge punctuation"]
    k1_l4 = {l4(r["text"]) for r in d1}
    nfd_l4_miss = sum(1 for r in nfd_rows if l4(r["text"]) not in k1_l4)
    # Shift scenario: the day-1 cache was filled by sources that only ever sent
    # NFC (drop day-1 NFD rows), then one source starts sending NFD on day 2.
    raw1_nfc_only = {r["text"] for r in d1 if r["variant"] != "nfd"}
    accented = [r for r in nfd_rows]  # every nfd row is accented by construction
    shift_raw_miss = sum(1 for r in accented if r["text"] not in raw1_nfc_only)
    out["nfd_shift_scenario"] = {"day2_nfd_known_rows": len(accented), "raw_key_misses": shift_raw_miss,
                                 "l4_key_misses": sum(1 for r in accented if l4(r["text"]) not in k1_l4)}
    out["nfd"] = {"day2_nfd_rows": len(nfd_rows), "raw_key_misses": nfd_raw_miss, "l4_key_misses": nfd_l4_miss,
                  "example_nfc_vs_nfd_equal_as_bytes": "débité" == "débité"}
    # What normalization cannot fix: composition of the L4 distinct set by the
    # variant that produced its first occurrence.
    first = {}
    for r in d1:
        first.setdefault(l4(r["text"]), r["variant"])
    out["day1_l4_distinct_by_first_variant"] = Counter(first.values())
    out["day1_variant_mix"] = Counter(r["variant"] for r in d1)
    out["day1_lang_mix"] = Counter(r["lang"] for r in d1)
    out["day2_source_mix"] = Counter(r["source"] for r in d2)
    save("dedupe", out)


if __name__ == "__main__":
    main()
