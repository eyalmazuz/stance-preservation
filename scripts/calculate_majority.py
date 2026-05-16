# /// script
# dependencies = [
#   "polars",
#   "nltk",
# ]
# ///

import argparse
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

import polars as pl
from nltk.stem.snowball import SnowballStemmer

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

HEBREW_PREFIXES = {"ה", "ב", "ל", "ו", "כ", "מ"}
HEBREW_SUFFIXES = {"יות", "ים", "ות", "ה", "ת"}
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
ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "between",
    "by", "for", "from", "in", "into", "is", "its", "of", "on", "or", "over",
    "that", "the", "their", "these", "this", "those", "to", "under", "was",
    "were", "with", "within", "without",
}

TOKEN_STRIP_CHARS = ".,;:!?\"'()[]{}<>"
TOPIC_JACCARD_THRESHOLD = 0.4
TOPIC_PHRASE_FUZZY_THRESHOLD = 0.80

english_stemmer = SnowballStemmer("english")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate normalized majority topic/stance labels with row-local voting."
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV path")
    parser.add_argument("output_csv", type=Path, help="Output CSV path")
    parser.add_argument(
        "--tie-break-prefix",
        type=str,
        default="annotator_B",
        help="Column prefix for tie-breaking. Default: annotator_B.",
    )
    parser.add_argument(
        "--annotators",
        type=str,
        help="Comma-separated list of annotator prefixes to include in majority voting (e.g. annotator_A,Gemini). If omitted, all available annotators are used.",
    )
    return parser.parse_args()


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return str(value).strip().lower() in {"", "nan", "null", "none"}


def contains_hebrew(text: str) -> bool:
    return any("\u0590" <= char <= "\u05ff" for char in text)


def normalize_stance(value: object) -> str | None:
    if is_missing(value):
        return None
    cleaned = str(value).strip()
    
    # Strip common bracket wrappers some annotators used
    cleaned_no_brackets = cleaned.strip("<>[]{}()").strip()
    
    # Check for empty or explicitly invalid strings
    if not cleaned_no_brackets or cleaned_no_brackets in {"ריק", "העמדה", "blank", "empty"}:
        return None

    # Exact matches for standardized English output
    if cleaned in {"Favor", "Against", "Neutral"}:
        return cleaned
    
    # Try mapping
    mapped = STANCE_MAP.get(cleaned.lower()) or STANCE_MAP.get(cleaned) or STANCE_MAP.get(cleaned_no_brackets)
    if mapped is not None:
        return mapped
    return cleaned


def stem_hebrew_token(token: str) -> str:
    cleaned = token
    # Strip prefix
    if len(cleaned) > 3:
        for prefix in HEBREW_PREFIXES:
            if cleaned.startswith(prefix) and len(cleaned) - 1 >= 3:
                cleaned = cleaned[1:]
                break
    # Strip suffix
    for suffix in HEBREW_SUFFIXES:
        if cleaned.endswith(suffix) and len(cleaned) - len(suffix) >= 3:
            cleaned = cleaned[:-len(suffix)]
            break
    return cleaned


def extract_normalized_tokens(text: str) -> list[str]:
    is_heb = contains_hebrew(text)
    stopwords = HEBREW_STOPWORDS if is_heb else ENGLISH_STOPWORDS

    tokens = []
    for word in text.casefold().split():
        cleaned = word.strip(TOKEN_STRIP_CHARS)
        # Remove internal punctuation keeping only letters/numbers
        cleaned = re.sub(r"[^\w\u0590-\u05FF]", "", cleaned)
        if not cleaned:
            continue
        
        if is_heb:
            # Handle standard hebrew stopword removal before prefix stripping if needed,
            # but usually it's better to check stopword exact match first.
            if cleaned in stopwords:
                continue
            stemmed = stem_hebrew_token(cleaned)
            if stemmed not in stopwords:
                tokens.append(stemmed)
        else:
            if cleaned in stopwords:
                continue
            stemmed = english_stemmer.stem(cleaned)
            tokens.append(stemmed)
            
    return tokens


def get_normalized_string(text: str) -> str:
    return " ".join(extract_normalized_tokens(text))


def topics_match(norm_left: str, norm_right: str) -> bool:
    if not norm_left or not norm_right:
        return False
    if norm_left == norm_right:
        return True

    left_tokens = set(norm_left.split())
    right_tokens = set(norm_right.split())

    if left_tokens and right_tokens:
        union = left_tokens | right_tokens
        intersection = left_tokens & right_tokens
        jaccard = len(intersection) / len(union) if union else 0.0
        
        if jaccard >= TOPIC_JACCARD_THRESHOLD or left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
            return True

    fuzzy_score = SequenceMatcher(None, norm_left, norm_right).ratio()
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


def choose_representative(original_values: list[str]) -> str:
    """Choose the most frequent original string as the representative for the cluster."""
    counts = Counter(original_values)
    # Get the most common, break ties by choosing the shortest string for simplicity
    max_count = max(counts.values())
    candidates = [val for val, cnt in counts.items() if cnt == max_count]
    return min(candidates, key=len)


def canonicalize_and_vote_topics(
    original_values: list[str | None], 
    tie_break_index: int | None
) -> tuple[list[str | None], str | None]:
    """
    Returns:
    - List of updated values where each valid original is replaced by its cluster representative.
    - The majority representative string (or None).
    """
    valid_indexes = [i for i, val in enumerate(original_values) if not is_missing(val)]
    if not valid_indexes:
        return [None] * len(original_values), None

    normalized_map = {i: get_normalized_string(str(original_values[i])) for i in valid_indexes}
    
    parent = list(range(len(original_values)))
    for left_index, right_index in combinations(valid_indexes, 2):
        if topics_match(normalized_map[left_index], normalized_map[right_index]):
            union(parent, left_index, right_index)

    # Group original values by their root
    clusters: dict[int, list[str]] = defaultdict(list)
    for i in valid_indexes:
        root = find(parent, i)
        clusters[root].append(str(original_values[i]).strip())

    # Find representatives for each cluster
    representatives = {root: choose_representative(vals) for root, vals in clusters.items()}
    
    # Map back to updated list
    updated_values = [None] * len(original_values)
    cluster_votes = Counter()
    
    for i in valid_indexes:
        root = find(parent, i)
        rep = representatives[root]
        updated_values[i] = rep
        cluster_votes[root] += 1

    # Majority logic
    max_votes = max(cluster_votes.values())
    winning_roots = {r for r, cnt in cluster_votes.items() if cnt == max_votes}
    
    majority_value = None
    if len(winning_roots) > 1 and tie_break_index is not None and tie_break_index in valid_indexes:
        tb_root = find(parent, tie_break_index)
        if tb_root in winning_roots:
            majority_value = representatives[tb_root]
            
    if majority_value is None:
        # Just pick the first winning root if tie-breaker didn't resolve it or wasn't in winners
        majority_value = representatives[next(iter(winning_roots))]

    return updated_values, majority_value


def process_stance_row(
    original_values: list[str | None],
    tie_break_index: int | None
) -> tuple[list[str | None], str | None]:
    
    normalized_values = [normalize_stance(val) for val in original_values]
    valid_indexes = [i for i, val in enumerate(normalized_values) if val is not None]
    
    if not valid_indexes:
        return [None] * len(original_values), None

    counts = Counter(normalized_values[i] for i in valid_indexes)
    max_count = max(counts.values())
    winners = {val for val, cnt in counts.items() if cnt == max_count}
    
    majority_val = None
    if len(winners) > 1 and tie_break_index is not None and tie_break_index in valid_indexes:
        tb_val = normalized_values[tie_break_index]
        if tb_val in winners:
            majority_val = tb_val
            
    if majority_val is None:
        # Pick arbitrary winner
        majority_val = next(iter(winners))
        
    return normalized_values, majority_val


def find_source_label_columns(columns: list[str], suffix: str, allowed_prefixes: list[str] | None = None) -> list[str]:
    cols = [column for column in columns if column.endswith(f"_{suffix}") and not column.startswith("majority_")]
    if allowed_prefixes:
        cols = [
            col for col in cols
            if any(col.startswith(f"{prefix}_") for prefix in allowed_prefixes)
        ]
    return cols


def main() -> None:
    args = parse_args()

    allowed_prefixes = [p.strip() for p in args.annotators.split(",")] if args.annotators else None

    df = pl.read_csv(args.input_csv)
    columns = df.columns
    
    updated_series: dict[str, list[str | None]] = defaultdict(list)
    majority_series: dict[str, list[str | None]] = {}
    
    sum_topic_cols = find_source_label_columns(columns, "summary_topic", allowed_prefixes)
    art_topic_cols = find_source_label_columns(columns, "article_topic", allowed_prefixes)
    sum_stance_cols = find_source_label_columns(columns, "summary_stance", allowed_prefixes)
    art_stance_cols = find_source_label_columns(columns, "article_stance", allowed_prefixes)

    for col in sum_topic_cols + art_topic_cols + sum_stance_cols + art_stance_cols:
        updated_series[col] = []

    tb_sum_topic_idx = sum_topic_cols.index(f"{args.tie_break_prefix}_summary_topic") if f"{args.tie_break_prefix}_summary_topic" in sum_topic_cols else None
    tb_art_topic_idx = art_topic_cols.index(f"{args.tie_break_prefix}_article_topic") if f"{args.tie_break_prefix}_article_topic" in art_topic_cols else None
    tb_sum_stance_idx = sum_stance_cols.index(f"{args.tie_break_prefix}_summary_stance") if f"{args.tie_break_prefix}_summary_stance" in sum_stance_cols else None
    tb_art_stance_idx = art_stance_cols.index(f"{args.tie_break_prefix}_article_stance") if f"{args.tie_break_prefix}_article_stance" in art_stance_cols else None

    maj_sum_topic_list = []
    maj_art_topic_list = []
    maj_sum_stance_list = []
    maj_art_stance_list = []

    for row in df.iter_rows(named=True):
        # Topics
        sum_topic_vals = [row[c] for c in sum_topic_cols]
        upd_sum_topic, maj_sum_topic = canonicalize_and_vote_topics(sum_topic_vals, tb_sum_topic_idx)

        art_topic_vals = [row[c] for c in art_topic_cols]
        upd_art_topic, maj_art_topic = canonicalize_and_vote_topics(art_topic_vals, tb_art_topic_idx)

        # Cross-Side Reconciliation
        if maj_sum_topic and maj_art_topic:
            norm_sum = get_normalized_string(maj_sum_topic)
            norm_art = get_normalized_string(maj_art_topic)
            
            if topics_match(norm_sum, norm_art):
                # Pick the longer one as the global representative for the row
                global_rep = maj_sum_topic if len(maj_sum_topic) >= len(maj_art_topic) else maj_art_topic
                norm_global = get_normalized_string(global_rep)

                # Realign all cells across both sides that matched this topic
                for i in range(len(upd_sum_topic)):
                    if upd_sum_topic[i] and topics_match(get_normalized_string(upd_sum_topic[i]), norm_global):
                        upd_sum_topic[i] = global_rep
                        
                for i in range(len(upd_art_topic)):
                    if upd_art_topic[i] and topics_match(get_normalized_string(upd_art_topic[i]), norm_global):
                        upd_art_topic[i] = global_rep
                
                maj_sum_topic = global_rep
                maj_art_topic = global_rep

        # Stances
        sum_stance_vals = [row[c] for c in sum_stance_cols]
        upd_sum_stance, maj_sum_stance = process_stance_row(sum_stance_vals, tb_sum_stance_idx)

        art_stance_vals = [row[c] for c in art_stance_cols]
        upd_art_stance, maj_art_stance = process_stance_row(art_stance_vals, tb_art_stance_idx)

        # Append
        for c, v in zip(sum_topic_cols, upd_sum_topic): updated_series[c].append(v)
        for c, v in zip(art_topic_cols, upd_art_topic): updated_series[c].append(v)
        for c, v in zip(sum_stance_cols, upd_sum_stance): updated_series[c].append(v)
        for c, v in zip(art_stance_cols, upd_art_stance): updated_series[c].append(v)

        maj_sum_topic_list.append(maj_sum_topic)
        maj_art_topic_list.append(maj_art_topic)
        maj_sum_stance_list.append(maj_sum_stance)
        maj_art_stance_list.append(maj_art_stance)

    majority_series["majority_summary_topic"] = maj_sum_topic_list
    majority_series["majority_article_topic"] = maj_art_topic_list
    majority_series["majority_summary_stance"] = maj_sum_stance_list
    majority_series["majority_article_stance"] = maj_art_stance_list

    result = df
    for col, values in updated_series.items():
        result = result.with_columns(pl.Series(col, values))
        
    for col, values in majority_series.items():
        result = result.with_columns(pl.Series(col, values))
    
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(args.output_csv)

    print(f"Read {len(df)} rows from {args.input_csv}")
    print("Topic grouping scope: row-local normalized Jaccard/Fuzzy with cross-side reconciliation")
    print(f"Tie-break prefix: {args.tie_break_prefix}")
    if allowed_prefixes:
        print(f"Filtered annotators: {', '.join(allowed_prefixes)}")
    print(f"majority_summary_topic: {len(sum_topic_cols)} source columns updated")
    print(f"majority_article_topic: {len(art_topic_cols)} source columns updated")
    print(f"majority_summary_stance: {len(sum_stance_cols)} source columns updated")
    print(f"majority_article_stance: {len(art_stance_cols)} source columns updated")
    print(f"Wrote output to {args.output_csv}")


if __name__ == "__main__":
    main()
