"""Cache-key normalization, one level at a time so each level's effect is measurable.

L0  raw bytes as received
L1  collapse Unicode whitespace (str.split() also splits on U+00A0)
L2  L1 + NFKC — precomposes NFD accents, folds compatibility forms
L3  L2 + casefold — not lower(): casefold also handles ß → ss
L4  L3 + strip leading/trailing punctuation runs (inner apostrophes survive)
"""
from __future__ import annotations

import re
import unicodedata

_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")


def l1(s: str) -> str:
    return " ".join(s.split())


def l2(s: str) -> str:
    return unicodedata.normalize("NFKC", l1(s))


def l3(s: str) -> str:
    return l2(s).casefold()


def l4(s: str) -> str:
    return _EDGE_PUNCT.sub("", l3(s)).strip()


LEVELS = {"L0 raw": lambda s: s, "L1 whitespace": l1, "L2 +NFKC": l2, "L3 +casefold": l3, "L4 +edge punctuation": l4}

key = l4  # the key the pipeline uses
