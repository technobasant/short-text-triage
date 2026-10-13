"""Embedding throughput: distinct normalized strings vs every row."""
from __future__ import annotations

import numpy as np

from .embed import encode, machine, model, timed
from .io import OUT, rows, save
from .normalize import key


def main() -> None:
    d1 = rows("day1")
    distinct = sorted({key(r["text"]) for r in d1})
    raw_rows = [r["text"] for r in d1]
    model()
    encode(["warm up"] * 64)  # first call pays lazy init; not part of any timing

    res = {"machine": machine(), "runs": 3, "results": []}
    for label, texts in (("day-1 distinct L4 keys", distinct), ("day-1 rows", raw_rows)):
        bs = 64
        vecs, wmed, walls, cmed, cpus = timed(lambda: encode(texts, batch_size=bs))
        res["results"].append({"input": label, "n": len(texts), "batch_size": bs,
                               "median_wall_s": round(wmed, 2), "wall_runs_s": [round(t, 2) for t in walls],
                               "median_cpu_s": round(cmed, 2), "cpu_runs_s": [round(t, 2) for t in cpus],
                               "texts_per_wall_s": round(len(texts) / wmed), "cpu_ms_per_1k_texts": round(1000 * cmed / len(texts) * 1000)})
    res["load_avg_end"] = [round(v, 1) for v in __import__("os").getloadavg()]
    save("embed_throughput", res)
    np.save(OUT / "emb_day1_distinct.npy", encode(distinct))
    (OUT / "day1_distinct.txt").write_text("\n".join(distinct), encoding="utf-8")


if __name__ == "__main__":
    main()
