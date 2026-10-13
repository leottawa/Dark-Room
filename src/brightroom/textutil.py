"""Text helpers shared by the extraction audit and the draft verifier."""

from __future__ import annotations

import re
import unicodedata

_NUMBER = re.compile(r"(\d+(?:[.,]\d+)*)(k\b)?", re.IGNORECASE)
_SPACED_THOUSANDS = re.compile(r"(?<![\d.,])(\d{1,3}) (\d{3})(?![\d.,])")
_PUNCTUATION = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalise(text: str) -> str:
    """Fold case, quotes, dashes and whitespace so a copied quote matches its source."""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION)
    return " ".join(text.casefold().split())


def quote_in(quote: str, source: str) -> bool:
    """True when the quote is a verbatim span of the source, ignoring case and outer punctuation."""
    needle = normalise(quote).strip(" \"'.,;:!?…")
    return bool(needle) and needle in normalise(source)


def _parse(token: str) -> float:
    parts = re.split(r"[.,]", token)
    if len(parts) == 1:
        return float(token)
    *head, last = parts
    grouped = len(last) == 3 and len(head[0]) <= 3 and all(len(p) == 3 for p in head[1:])
    if grouped:  # 1,500 or 1.500
        return float("".join(parts))
    return float("".join(head) + "." + last)  # 12.50, 4,5 or 1,500.50


def numbers_in(text: str, loose: bool = False) -> set[float]:
    """Every number written in digits. With loose=True, "1 500" also counts as 1500."""
    found = {_parse(token) * (1000 if k else 1) for token, k in _NUMBER.findall(text)}
    if loose:
        found |= {float(a + b) for a, b in _SPACED_THOUSANDS.findall(text)}
    return found


_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_WORDS |= {"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "hundred": 100, "thousand": 1000}


def contains_number(value: float, text: str) -> bool:
    """True when the value is written in the text, in digits or as a plain number word."""
    written = numbers_in(text, loose=True) | {float(_WORDS[w]) for w in re.findall(r"[a-z]+", normalise(text)) if w in _WORDS}
    return any(abs(value - n) < 0.005 for n in written)
