import re

from difflib import SequenceMatcher

HEBREW_NIQQUD_RE = re.compile(r"[\u0591-\u05c7]")
HEBREW_TOKEN_RE = re.compile(r"^[\u0590-\u05ff]+$")
LEXICAL_TOKEN_RE = re.compile(r"[\u0590-\u05ff]+|[A-Za-z0-9]+")
HEBREW_FINAL_LETTERS = str.maketrans(
    {
        "\u05da": "\u05db",
        "\u05dd": "\u05de",
        "\u05df": "\u05e0",
        "\u05e3": "\u05e4",
        "\u05e5": "\u05e6",
    }
)
HEBREW_PROCLITICS = (
    "\u05d5\u05db\u05e9",
    "\u05d5\u05e9\u05d4",
    "\u05e9\u05d1",
    "\u05e9\u05d4",
    "\u05db\u05e9",
    "\u05d5\u05d4",
    "\u05d5\u05d1",
    "\u05d5\u05dc",
    "\u05d5\u05de",
    "\u05d1",
    "\u05dc",
    "\u05db",
    "\u05de",
    "\u05d4",
    "\u05d5",
    "\u05e9",
)


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


def normalize_hebrew_token(token: str) -> list[str]:
    token = HEBREW_NIQQUD_RE.sub("", token).translate(HEBREW_FINAL_LETTERS)
    if not token:
        return []
    if not HEBREW_TOKEN_RE.fullmatch(token) or len(token) <= 3:
        return [token.lower()]

    for prefix in HEBREW_PROCLITICS:
        stem = token[len(prefix) :]
        if token.startswith(prefix) and len(stem) >= 3:
            return [prefix, stem]

    return [token]


def hebrew_morph_normalize(text: str) -> str:
    normalized_tokens: list[str] = []
    for token in LEXICAL_TOKEN_RE.findall(text):
        normalized_tokens.extend(normalize_hebrew_token(token))
    return " ".join(normalized_tokens)
