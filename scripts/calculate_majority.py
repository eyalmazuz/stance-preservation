import argparse
import re

from collections import Counter
from collections.abc import Callable
from difflib import SequenceMatcher
from pathlib import Path

import polars as pl

STANCE_MAP = {
    "בעד": "Favor",
    "תומך": "Favor",
    "favor": "Favor",
    "against": "Against",
    "נגד": "Against",
    "נוגד": "Against",
    "<נגד>": "Against",
    "neutral": "Neutral",
    "נייטרלי": "Neutral",
    "ניטרלי": "Neutral",
    "עמדה נייטרלית": "Neutral",
    "<נייטרלי>": "Neutral",
}

TARGET_SUFFIXES = (
    "summary_topic",
    "summary_stance",
    "article_topic",
    "article_stance",
)

HEBREW_PREFIXES = {"ה", "ב", "ל", "ו"}
HEBREW_STOPWORDS = {
    "של",
    "את",
    "על",
    "עם",
    "כי",
    "הוא",
    "היא",
    "הם",
    "הן",
    "בתוך",
    "סרט",
    "סדרה",
}
TOPIC_JACCARD_THRESHOLD = 0.4
TOPIC_TOKEN_FUZZY_THRESHOLD = 0.80
TOPIC_PHRASE_FUZZY_THRESHOLD = 0.72
HEBREW_LEADING_PREFIXES = ("ה", "ו", "ב", "ל", "כ", "מ")
HEBREW_TRAILING_SUFFIXES = ("יות", "ים", "ות", "ה", "ת")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate majority topic/stance labels from dynamically discovered annotator/model columns."
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV path")
    parser.add_argument("output_csv", type=Path, help="Output CSV path")
    return parser.parse_args()


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return str(value).strip().lower() in {"", "nan", "null", "none"}


def normalize_stance(value: object) -> str | None:
    if is_missing(value):
        return None

    cleaned = str(value).strip()
    if cleaned in {"Favor", "Against", "Neutral"}:
        return cleaned

    mapped = STANCE_MAP.get(cleaned.lower())
    if mapped is not None:
        return mapped

    mapped = STANCE_MAP.get(cleaned)
    if mapped is not None:
        return mapped

    return cleaned


def normalize_topic(value: object) -> str | None:
    if is_missing(value):
        return None
    return str(value).strip()


def find_label_columns(columns: list[str], suffix: str) -> list[str]:
    return [column for column in columns if column.endswith(f"_{suffix}")]


def find_annotator_b_column(columns: list[str], suffix: str) -> str | None:
    target = f"annotator_B_{suffix}"
    return target if target in columns else None


def get_keywords(text: str) -> set[str]:
    words = text.lower().split()
    keywords: set[str] = set()

    for word in words:
        token = word.strip(".,;:!?\"'()[]{}<>")
        if len(token) > 3 and token[0] in HEBREW_PREFIXES:
            token = token[1:]
        if len(token) > 1 and token not in HEBREW_STOPWORDS:
            keywords.add(token)

    return keywords


def clean_token(token: str) -> str:
    return re.sub(r"[^\w\u0590-\u05FF]", "", token).strip()


def singularize_token(token: str) -> str:
    for suffix in HEBREW_TRAILING_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def strip_prefix_token(token: str) -> str:
    if len(token) <= 3:
        return token

    for prefix in HEBREW_LEADING_PREFIXES:
        if token.startswith(prefix) and len(token) - 1 >= 3:
            return token[1:]

    return token


def token_variants(token: str) -> set[str]:
    cleaned = clean_token(token)
    if not cleaned:
        return set()

    singular = singularize_token(cleaned)
    stripped = strip_prefix_token(cleaned)
    stripped_singular = singularize_token(stripped)

    return {variant for variant in {cleaned, singular, stripped, stripped_singular} if len(variant) >= 2}


def topic_variants(text: str) -> set[str]:
    variants: set[str] = set()
    for token in get_keywords(text):
        variants |= token_variants(token)
    return variants


def topics_match(left: str, right: str) -> bool:
    left_keywords = get_keywords(left)
    right_keywords = get_keywords(right)

    if left_keywords and right_keywords:
        union = left_keywords | right_keywords
        intersection = left_keywords & right_keywords
        jaccard = len(intersection) / len(union) if union else 0.0
        if (
            jaccard >= TOPIC_JACCARD_THRESHOLD
            or left_keywords.issubset(right_keywords)
            or right_keywords.issubset(left_keywords)
        ):
            return True

    left_variants = topic_variants(left)
    right_variants = topic_variants(right)

    if left_variants & right_variants:
        return True

    for left_variant in left_variants:
        for right_variant in right_variants:
            if left_variant in right_variant or right_variant in left_variant:
                return True
            if SequenceMatcher(None, left_variant, right_variant).ratio() >= TOPIC_TOKEN_FUZZY_THRESHOLD:
                return True

    fuzzy_score = SequenceMatcher(None, left.casefold(), right.casefold()).ratio()
    return fuzzy_score >= TOPIC_PHRASE_FUZZY_THRESHOLD


def build_topic_canonical_map(frame: pl.DataFrame, topic_columns: list[str]) -> dict[str, str]:
    canonical_map: dict[str, str] = {}
    canonical_topics: list[str] = []

    for row in frame.select(topic_columns).iter_rows():
        for value in row:
            topic = normalize_topic(value)
            if topic is None or topic in canonical_map:
                continue

            matched = next((candidate for candidate in canonical_topics if topics_match(topic, candidate)), None)
            if matched is None:
                canonical_topics.append(topic)
                canonical_map[topic] = topic
            else:
                canonical_map[topic] = matched

    return canonical_map


def majority_vote(values: list[str | None], tie_break_value: str | None = None) -> str | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None

    counts = Counter(valid_values)
    max_count = max(counts.values())
    winners = {value for value, count in counts.items() if count == max_count}

    if len(winners) > 1 and tie_break_value in winners:
        return tie_break_value

    for value in valid_values:
        if value in winners:
            return value

    return None


def build_majority_column(
    frame: pl.DataFrame,
    source_columns: list[str],
    normalizer: Callable[[object], str | None],
    tie_break_column: str | None = None,
) -> list[str | None]:
    selected_columns = source_columns.copy()
    if tie_break_column is not None and tie_break_column not in selected_columns:
        selected_columns.append(tie_break_column)

    rows = frame.select(selected_columns).iter_rows(named=True)
    majority_values: list[str | None] = []

    for row in rows:
        normalized_values = [normalizer(row[column]) for column in source_columns]
        tie_break_value = normalizer(row[tie_break_column]) if tie_break_column is not None else None
        majority_values.append(majority_vote(normalized_values, tie_break_value))

    return majority_values


def main() -> None:
    args = parse_args()

    df = pl.read_csv(args.input_csv)
    columns = df.columns

    missing_suffixes = [suffix for suffix in TARGET_SUFFIXES if not find_label_columns(columns, suffix)]
    if missing_suffixes:
        missing_str = ", ".join(missing_suffixes)
        raise ValueError(f"Could not find any columns for: {missing_str}")

    topic_columns = [
        column for suffix in ("summary_topic", "article_topic") for column in find_label_columns(columns, suffix)
    ]
    topic_canonical_map = build_topic_canonical_map(df, topic_columns)

    def normalize_topic_group(value: object) -> str | None:
        topic = normalize_topic(value)
        if topic is None:
            return None
        return topic_canonical_map.get(topic, topic)

    result = df
    majority_specs = (
        ("summary_topic", "majority_summary_topic", normalize_topic_group),
        ("summary_stance", "majority_summary_stance", normalize_stance),
        ("article_topic", "majority_article_topic", normalize_topic_group),
        ("article_stance", "majority_article_stance", normalize_stance),
    )

    for suffix, output_column, normalizer in majority_specs:
        source_columns = find_label_columns(columns, suffix)
        tie_break_column = find_annotator_b_column(columns, suffix)
        majority_values = build_majority_column(
            result,
            source_columns,
            normalizer,
            tie_break_column=tie_break_column,
        )
        result = result.with_columns(pl.Series(output_column, majority_values))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(args.output_csv)

    print(f"Read {len(df)} rows from {args.input_csv}")
    for suffix, output_column, _ in majority_specs:
        source_columns = find_label_columns(columns, suffix)
        print(f"{output_column}: {len(source_columns)} source columns")
    print(f"Wrote output to {args.output_csv}")


if __name__ == "__main__":
    main()
