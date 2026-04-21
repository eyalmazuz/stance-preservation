import argparse

import polars as pl

from scipy.stats import kendalltau, pearsonr, spearmanr
from tqdm.auto import tqdm

from src.models import BleuScorer, EmbeddingScorer, EMDScorer, LLMScorer, NLIScorer, RougeScorer, TfIdfScorer
from src.utils.data_utils import process_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Path to the CSV file containing the data.")
    parser.add_argument(
        "--label-prefix",
        type=str,
        choices=["GPT", "Gemini", "annotator_A", "annotator_B", "majority"],
        default="majority",
        help="Which annotation to use as the ground truth.",
    )
    parser.add_argument(
        "--language", type=str, choices=["he", "en"], default="he", help="Which langauge the dataset is."
    )
    parser.add_argument(
        "--aggregate-level",
        type=str,
        choices=["sentence", "article"],
        default="article",
        help=(
            "Which level to calcualte the data,"
            "article means we treat the article as a single unit."
            "Sentence means we split the article into sentences"
            "and calculate the metrics at the sentence level, then we aggregate the results."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["bleu", "rouge1", "rouge2", "rougeL", "tf-idf", "emb", "llm", "nli", "emd"],
        default="bleu",
        help="Which model to use to calculate correlations.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-large",
        help="Which embedding model to use when using text embedding baseline.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="Which llm model to use when using text llm baseline.",
    )
    parser.add_argument(
        "--nli-model",
        type=str,
        default="joeddav/xlm-roberta-large-xnli",
        help="Which NLI model to use when using text NLI baseline.",
    )
    parser.add_argument(
        "--matching-model",
        type=str,
        default="intfloat/multilingual-e5-large-instruct",
        help="Which sentence matching model to use when using text NLI baseline.",
    )
    parser.add_argument(
        "--topic-model",
        type=str,
        default="dicta-il/dictalm2.0",
        help="Which sentence matching model to use when using text NLI baseline.",
    )
    parser.add_argument(
        "--stance-model",
        type=str,
        default="./models/stance_detection",
        help="Which stance model to use when using text NLI baseline.",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.0,
        help="Whether to filter pairs based on stance model entropy",
    )
    parser.add_argument(
        "--use-topic-filtering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to filter pairs based on if the topic match",
    )
    parser.add_argument(
        "--use-soft-topic-filtering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to filter pairs based on if the topic similarly match",
    )
    parser.add_argument(
        "--use-dist-topic-score",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to add topic distance score to the final scoring",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pl.read_csv(args.input_file)

    if args.model == "bleu":
        scorer = BleuScorer()
    elif args.model.startswith("rouge"):
        scorer = RougeScorer(args.model)
    elif args.model == "tf-idf":
        scorer = TfIdfScorer()
    elif args.model == "emb":
        scorer = EmbeddingScorer(args.embedding_model)
    elif args.model == "llm":
        scorer = LLMScorer(args.llm_model)
    elif args.model == "nli":
        scorer = NLIScorer(
            args.nli_model,
            args.aggregate_level,
            args.language,
        )
    elif args.model == "emd":
        scorer = EMDScorer(
            args.matching_model,
            args.topic_model,
            args.stance_model,
            args.aggregate_level,
            args.language,
            args.entropy_threshold,
            args.use_topic_filtering,
            args.use_soft_topic_filtering,
            args.use_dist_topic_score,
        )
    else:
        raise ValueError("Not implemented yet")

    scores: list[float] = []
    preds: list[float] = []

    data = process_data(df, args.label_prefix)
    for pair in tqdm(data):
        if args.aggregate_level == "sentence" or args.model == "nli":
            hypotheses = pair.summary_data
            references = pair.article_data
        else:
            hypotheses = pair.summary
            references = pair.article

        scores.append(pair.score)
        pred = scorer.score(hypotheses, references)
        preds.append(pred)

    for name, corr in [("Pearson", pearsonr), ("Spearman", spearmanr), ("Kendall", kendalltau)]:
        stat, pvalue = corr(preds, scores)
        print(f"{name}- corr: {stat:.3f}, p-value {pvalue:.3f}")


if __name__ == "__main__":
    main()
