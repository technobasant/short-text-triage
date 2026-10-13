"""Local multilingual-e5-small on CPU. One model load per process."""
from __future__ import annotations

import os

# Under a loaded machine, tokenizer threads and torch threads fight each other.
# Pin both so the measurement is about the model, not the scheduler.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import platform
import subprocess
import time
from functools import lru_cache

import numpy as np

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


@lru_cache(maxsize=1)
def model():
    import torch
    torch.set_num_threads(int(os.environ.get("TRIAGE_THREADS", "4")))
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL, revision=REVISION, device="cpu")


def encode(texts: list[str], prefix: str = "query: ", batch_size: int = 64) -> np.ndarray:
    return model().encode([prefix + t for t in texts], batch_size=batch_size,
                          normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def machine() -> dict:
    import sentence_transformers, sklearn, torch, umap
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return {"chip": chip, "cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(), "python": platform.python_version(),
            "torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__,
            "scikit_learn": sklearn.__version__, "umap_learn": umap.__version__, "numpy": np.__version__,
            "model": MODEL, "revision": REVISION, "load_avg_1_5_15": [round(v, 1) for v in os.getloadavg()]}


def timed(fn, runs: int = 3):
    """Median wall-clock AND median process CPU time over `runs`.

    CPU time is the number to trust on a shared machine: wall-clock includes
    every second the scheduler gave to something else.
    """
    walls, cpus = [], []
    for _ in range(runs):
        w, c = time.perf_counter(), time.process_time()
        out = fn()
        walls.append(time.perf_counter() - w)
        cpus.append(time.process_time() - c)
    med = lambda v: sorted(v)[len(v) // 2]
    return out, med(walls), walls, med(cpus), cpus
