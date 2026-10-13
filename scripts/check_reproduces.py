"""Fail unless the deterministic half of the lab reproduces exactly.

`triage.corpus` (seeded) and `triage.dedupe` use only the standard library, so
CI can rerun them in seconds and require that out/results/dedupe.json matches
the committed copy exactly — every distinct-key count, cache hit rate and NFD
miss in section 1 of MEASUREMENTS.md.

The embedding, clustering and assignment steps are not rerun in CI: they need
torch and a ~470 MB model download, take minutes of CPU, and their timings are
hardware-specific by nature. Run ./run_all.sh locally for those.

    python -m triage.corpus && python -m triage.dedupe && python scripts/check_reproduces.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE = "out/results/dedupe.json"


def main() -> int:
    committed = json.loads(subprocess.run(["git", "show", f"HEAD:{FILE}"], cwd=ROOT,
                                          capture_output=True, text=True, check=True).stdout)
    rerun = json.loads((ROOT / FILE).read_text(encoding="utf-8"))
    if committed == rerun:
        print(f"{FILE} reproduces exactly from the seeded corpus")
        return 0
    for key in sorted(set(committed) | set(rerun)):
        if committed.get(key) != rerun.get(key):
            print(f"differs: {key}\n  committed: {committed.get(key)!r}\n  rerun:     {rerun.get(key)!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
