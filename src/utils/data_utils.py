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


def process_data(df: pl.DataFrame, prefix: str):
    grouped_df = df.group_by(["article", "summary"])

    text_data: list[TextPair] = []
    for (article, summary), data in tqdm(grouped_df):
        preservation_score = compute_instance_score(data, prefix)
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


def compute_instance_score(data: pl.DataFrame, prefix: str) -> float:
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
