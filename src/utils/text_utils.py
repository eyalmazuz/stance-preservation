import re

from difflib import SequenceMatcher


def norm_topic(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_jaccard(a: str, b: str) -> float:
    a_set = set(norm_topic(a).split())
    b_set = set(norm_topic(b).split())
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def char_ngrams(s: str, n: int = 3) -> set[str]:
    s = norm_topic(s).replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def char_jaccard(a: str, b: str, n: int = 3) -> float:
    a_set = char_ngrams(a, n)
    b_set = char_ngrams(b, n)
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_topic(a), norm_topic(b)).ratio()


def topics_match_soft(
    a: str,
    b: str,
    token_jaccard_threshold: float = 0.5,
    char_jaccard_threshold: float = 0.45,
    fuzzy_ratio_threshold: float = 0.82,
) -> bool:
    return (
        token_jaccard(a, b) >= token_jaccard_threshold
        or char_jaccard(a, b, n=3) >= char_jaccard_threshold
        or fuzzy_ratio(a, b) >= fuzzy_ratio_threshold
    )
