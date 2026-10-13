from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
RES = OUT / "results"


def rows(day: str) -> list[dict]:
    with open(OUT / f"{day}.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(name: str, obj) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(obj, indent=2, ensure_ascii=False))
