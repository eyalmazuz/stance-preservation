import argparse
import re

from collections import Counter, defaultdict
from collections.abc import Callable
from difflib import SequenceMatcher
from itertools import combinations
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
ENGLISH_TOPIC_STOPWORDS = {
    "article",
    "articles",
    "data",
    "detail",
    "details",
    "discussion",
    "film",
    "films",
    "movie",
    "movies",
    "performance",
    "performances",
    "publication",
    "review",
    "reviews",
    "series",
    "stories",
    "story",
    "studies",
    "study",
}

TOPIC_JACCARD_THRESHOLD = 0.4
TOPIC_TOKEN_FUZZY_THRESHOLD = 0.80
TOPIC_PHRASE_FUZZY_THRESHOLD = 0.72
HEBREW_LEADING_PREFIXES = ("ה", "ו", "ב", "ל", "כ", "מ")
HEBREW_TRAILING_SUFFIXES = ("יות", "ים", "ות", "ה", "ת")
TOKEN_STRIP_CHARS = ".,;:!?\"'()[]{}<>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate majority topic/stance labels without corpus-level topic leakage. "
            "Topic grouping is local to each row."
        )
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


def find_source_label_columns(columns: list[str], suffix: str) -> list[str]:
    return [column for column in columns if column.endswith(f"_{suffix}") and not column.startswith("majority_")]


def find_annotator_b_column(columns: list[str], suffix: str) -> str | None:
    target = f"annotator_B_{suffix}"
    return target if target in columns else None


def contains_hebrew(text: str) -> bool:
    return any("\u0590" <= char <= "\u05ff" for char in text)


def cleaned_keyword_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for word in text.casefold().split():
        token = word.strip(TOKEN_STRIP_CHARS)
        if contains_hebrew(token) and len(token) > 3 and token[0] in HEBREW_PREFIXES:
            token = token[1:]
        if len(token) > 1:
            tokens.append(token)
    return tokens


def get_keywords(text: str) -> set[str]:
    tokens = cleaned_keyword_tokens(text)
    filtered_tokens = []

    for token in tokens:
        stopwords = HEBREW_STOPWORDS if contains_hebrew(token) else ENGLISH_TOPIC_STOPWORDS
        if token not in stopwords:
            filtered_tokens.append(token)

    return set(filtered_tokens or tokens)


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


def topic_token_entries(text: str) -> list[tuple[str, set[str]]]:
    entries: list[tuple[str, set[str], bool]] = []

    for word in text.split():
        display_token = clean_token(word.strip(TOKEN_STRIP_CHARS))
        if not display_token:
            continue

        key = display_token.casefold()
        if contains_hebrew(key) and len(key) > 3 and key[0] in HEBREW_PREFIXES:
            key = key[1:]
        if len(key) <= 1:
            continue

        stopwords = HEBREW_STOPWORDS if contains_hebrew(key) else ENGLISH_TOPIC_STOPWORDS
        entries.append((display_token, token_variants(key), key in stopwords))

    non_stopword_entries = [(display, variants) for display, variants, is_stopword in entries if not is_stopword]
    if non_stopword_entries:
        return non_stopword_entries

    return [(display, variants) for display, variants, _ in entries]


def derive_topic_representative(topics: list[str]) -> str:
    topic_counts = Counter(topics)
    max_count = max(topic_counts.values())
    if max_count > 1:
        for topic in topics:
            if topic_counts[topic] == max_count:
                return topic

    topic_entries = [(topic, topic_token_entries(topic)) for topic in topics]
    topic_entries = [(topic, entries) for topic, entries in topic_entries if entries]
    if not topic_entries:
        return topics[0]

    variant_sets = [{variant for _, variants in entries for variant in variants} for _, entries in topic_entries]
    shared_variants = set.intersection(*variant_sets) if variant_sets else set()

    if shared_variants:
        _, source_entries = min(topic_entries, key=lambda item: (len(item[1]), len(item[0])))
        representative_tokens = [
            display for display, variants in source_entries if variants & shared_variants
        ]
        if representative_tokens:
            return " ".join(representative_tokens)

    most_common_topics = [topic for topic in topics if topic_counts[topic] == max_count]
    return min(most_common_topics, key=lambda topic: (len(topic_token_entries(topic)), len(topic)))


def topics_match(left: str, right: str) -> bool:
    if left.casefold() == right.casefold():
        return True

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
            if SequenceMatcher(None, left_variant, right_variant).ratio() >= TOPIC_TOKEN_FUZZY_THRESHOLD:
                return True

    fuzzy_score = SequenceMatcher(None, left.casefold(), right.casefold()).ratio()
    return fuzzy_score >= TOPIC_PHRASE_FUZZY_THRESHOLD


def find(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def union(parent: list[int], left: int, right: int) -> None:
    left_root = find(parent, left)
    right_root = find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def canonicalize_topic_values(values: list[object]) -> list[str | None]:
    topics = [normalize_topic(value) for value in values]
    valid_indexes = [index for index, topic in enumerate(topics) if topic is not None]

    if not valid_indexes:
        return [None] * len(values)

    parent = list(range(len(topics)))
    for left_index, right_index in combinations(valid_indexes, 2):
        left_topic = topics[left_index]
        right_topic = topics[right_index]
        if left_topic is not None and right_topic is not None and topics_match(left_topic, right_topic):
            union(parent, left_index, right_index)

    topic_groups: dict[int, list[str]] = defaultdict(list)
    for index in valid_indexes:
        topic = topics[index]
        if topic is not None:
            topic_groups[find(parent, index)].append(topic)

    representatives = {
        group_root: derive_topic_representative(group_topics) for group_root, group_topics in topic_groups.items()
    }

    return [
        representatives[find(parent, index)] if topic is not None else None
        for index, topic in enumerate(topics)
    ]


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


def build_row_local_topic_majority(
    frame: pl.DataFrame,
    source_columns: list[str],
    tie_break_column: str | None,
) -> list[str | None]:
    selected_columns = source_columns.copy()
    if tie_break_column is not None and tie_break_column not in selected_columns:
        selected_columns.append(tie_break_column)

    majority_values: list[str | None] = []
    for row in frame.select(selected_columns).iter_rows(named=True):
        canonical_values = canonicalize_topic_values([row[column] for column in source_columns])
        if tie_break_column is None:
            tie_break_value = None
        else:
            tie_break_index = source_columns.index(tie_break_column)
            tie_break_value = canonical_values[tie_break_index]
        majority_values.append(majority_vote(canonical_values, tie_break_value))

    return majority_values


def build_majority_column(
    frame: pl.DataFrame,
    source_columns: list[str],
    normalizer: Callable[[object], str | None],
    tie_break_column: str | None = None,
) -> list[str | None]:
    selected_columns = source_columns.copy()
    if tie_break_column is not None and tie_break_column not in selected_columns:
        selected_columns.append(tie_break_column)

    majority_values: list[str | None] = []
    for row in frame.select(selected_columns).iter_rows(named=True):
        normalized_values = [normalizer(row[column]) for column in source_columns]
        tie_break_value = normalizer(row[tie_break_column]) if tie_break_column is not None else None
        majority_values.append(majority_vote(normalized_values, tie_break_value))

    return majority_values


def validate_inputs(columns: list[str]) -> None:
    missing_suffixes = [suffix for suffix in TARGET_SUFFIXES if not find_source_label_columns(columns, suffix)]
    if missing_suffixes:
        missing_str = ", ".join(missing_suffixes)
        raise ValueError(f"Could not find any non-majority source columns for: {missing_str}")


def main() -> None:
    args = parse_args()

    df = pl.read_csv(args.input_csv)
    columns = df.columns
    validate_inputs(columns)

    result = df
    majority_specs = (
        ("summary_topic", "majority_summary_topic", normalize_topic),
        ("summary_stance", "majority_summary_stance", normalize_stance),
        ("article_topic", "majority_article_topic", normalize_topic),
        ("article_stance", "majority_article_stance", normalize_stance),
    )

    for suffix, output_column, normalizer in majority_specs:
        source_columns = find_source_label_columns(columns, suffix)
        tie_break_column = find_annotator_b_column(source_columns, suffix)

        if suffix.endswith("_topic"):
            majority_values = build_row_local_topic_majority(result, source_columns, tie_break_column)
        else:
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
    print("Topic grouping scope: row")
    for suffix, output_column, _ in majority_specs:
        source_columns = find_source_label_columns(columns, suffix)
        print(f"{output_column}: {len(source_columns)} source columns")
    print(f"Wrote output to {args.output_csv}")


if __name__ == "__main__":
    main()
