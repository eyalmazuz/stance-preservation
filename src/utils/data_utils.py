import re

from dataclasses import dataclass, field

import polars as pl

from tqdm.auto import tqdm


def split_into_sentences(text):
    """Split text into sentences."""
    if not isinstance(text, str):
        return []
    separators = r"[■|•.\n]"
    sentences = [sent.strip() for sent in re.split(separators, text) if sent.strip()]
    return sentences


@dataclass
class TextPair:
    article: str
    summary: str
    score: float
    article_data: list[dict[str, str]] = field(default_factory=list)
    summary_data: list[dict[str, str]] = field(default_factory=list)


STANCE_VALUE = {
    "Against": -1,
    "Neutral": 0,
    "Favor": 1,
}


def process_data(df: pl.DataFrame, prefix: str, score_method: str = "exact"):
    grouped_df = df.group_by(["article", "summary"], maintain_order=True)

    text_data: list[TextPair] = []
    for (article, summary), data in tqdm(grouped_df):
        preservation_score = compute_instance_score(data, prefix, score_method)
        article_data = build_sentence_data(data, prefix, "article")
        summary_data = build_sentence_data(data, prefix, "summary")
        pair = TextPair(article, summary, preservation_score, article_data, summary_data)

        text_data.append(pair)

    return text_data


def build_sentence_data(data: pl.DataFrame, prefix: str, sentence_type: str = "article"):
    if sentence_type == "article":
        sentences = data["best_match_sentences_from_article"].to_list()
        full_text = data["article"].to_list()
    else:
        sentences = data["sentence_in_summary"].to_list()
        full_text = data["summary"].to_list()
    topics = data[f"{prefix}_{sentence_type}_topic"].to_list()
    stances = data[f"{prefix}_{sentence_type}_stance"].to_list()

    sentences_data: list[dict[str, str]] = []
    for text, sentence, topic, stance in zip(full_text, sentences, topics, stances):
        sentence_data = {"full_text": text, "text": sentence, "topic": topic, "stance": stance}
        sentences_data.append(sentence_data)

    return sentences_data


def compute_instance_score(data: pl.DataFrame, prefix: str, score_method: str = "exact") -> float:
    if score_method == "exact":
        return compute_exact_conditional_score(data, prefix)
    if score_method == "conditional-soft":
        return compute_conditional_soft_score(data, prefix)
    if score_method == "coverage-soft":
        return compute_coverage_soft_score(data, prefix)
    raise ValueError(f"Invalid score method: {score_method}")


def compute_exact_conditional_score(data: pl.DataFrame, prefix: str) -> float:
    total = data.shape[0]
    matched = data.filter(
        (pl.col(f"{prefix}_summary_topic") == pl.col(f"{prefix}_article_topic"))
        & (pl.col(f"{prefix}_summary_stance") == pl.col(f"{prefix}_article_stance"))
    ).shape[0]
    diff = data.filter(pl.col(f"{prefix}_summary_topic") != pl.col(f"{prefix}_article_topic")).shape[0]

    if total - diff == 0:
        preservation_score = 0
    else:
        preservation_score = matched / (total - diff)

    return preservation_score


def compute_coverage_soft_score(data: pl.DataFrame, prefix: str) -> float:
    if data.is_empty():
        return 0.0

    total_score = 0.0
    for row in data.select(
        f"{prefix}_summary_topic",
        f"{prefix}_article_topic",
        f"{prefix}_summary_stance",
        f"{prefix}_article_stance",
    ).iter_rows(named=True):
        if row[f"{prefix}_summary_topic"] != row[f"{prefix}_article_topic"]:
            continue
        total_score += stance_similarity(row[f"{prefix}_summary_stance"], row[f"{prefix}_article_stance"])

    return total_score / data.shape[0]


def compute_conditional_soft_score(data: pl.DataFrame, prefix: str) -> float:
    if data.is_empty():
        return 0.0

    total_score = 0.0
    topic_matches = 0
    for row in data.select(
        f"{prefix}_summary_topic",
        f"{prefix}_article_topic",
        f"{prefix}_summary_stance",
        f"{prefix}_article_stance",
    ).iter_rows(named=True):
        if row[f"{prefix}_summary_topic"] != row[f"{prefix}_article_topic"]:
            continue
        topic_matches += 1
        total_score += stance_similarity(row[f"{prefix}_summary_stance"], row[f"{prefix}_article_stance"])

    if topic_matches == 0:
        return 0.0
    return total_score / topic_matches


def stance_similarity(summary_stance: str, article_stance: str) -> float:
    summary_value = STANCE_VALUE.get(summary_stance)
    article_value = STANCE_VALUE.get(article_stance)
    if summary_value is None or article_value is None:
        return 0.0
    return 1.0 - abs(summary_value - article_value) / 2.0
