import argparse
import os

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
        "--score-method",
        type=str,
        choices=["exact", "conditional-soft", "coverage-soft"],
        default="exact",
        help=(
            "Which gold score to use. exact is same-stance over same-topic pairs. "
            "conditional-soft is soft stance preservation over same-topic pairs. "
            "coverage-soft is coverage-weighted soft stance preservation over all summary pairs."
        ),
    )
    parser.add_argument(
        "--no-save-preds",
        action="store_true",
        default=False,
        help="Whether to save the predictions to the results file or not.",
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
        "--llm-prompt",
        type=str,
        default="./data/prompts/prediction_prompt.txt",
        help="Which prompt to use to classify with.",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=1,
        help="How many OpenAI LLM scoring calls to run in parallel within each article-summary pair.",
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
        help="Whether to filter pairs based on stance model entropy.",
    )
    parser.add_argument(
        "--use-topic-filtering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to filter pairs based on if the topic match.",
    )
    parser.add_argument(
        "--use-topic-mismatch-filtering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to filter BLEU pairs to gold topic mismatches only.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to print debug statistics during scoring.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use-soft-topic-filtering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to filter pairs based on if the topic similarly match.",
    )
    group.add_argument(
        "--use-dist-topic-score",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to add topic distance score to the final scoring.",
    )
    group.add_argument(
        "--use-weighted-emd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to weight the EMD score by the cosine sim.",
    )
    args = parser.parse_args()
    if args.use_topic_filtering and args.use_topic_mismatch_filtering:
        parser.error("--use-topic-filtering and --use-topic-mismatch-filtering are mutually exclusive.")
    if args.use_topic_mismatch_filtering and args.model != "bleu":
        parser.error("--use-topic-mismatch-filtering is currently supported only with --model bleu.")
    return args


def main():
    args = parse_args()
    df = pl.read_csv(args.input_file)

    bleu_topic_diagnostic = args.model == "bleu" and (args.use_topic_filtering or args.use_topic_mismatch_filtering)

    if args.model == "bleu":
        bleu_topic_filter = None
        if args.use_topic_filtering:
            bleu_topic_filter = "match"
        elif args.use_topic_mismatch_filtering:
            bleu_topic_filter = "mismatch"
        scorer = BleuScorer(topic_filter=bleu_topic_filter)
    elif args.model.startswith("rouge"):
        scorer = RougeScorer(args.model)
    elif args.model == "tf-idf":
        scorer = TfIdfScorer()
    elif args.model == "emb":
        scorer = EmbeddingScorer(args.embedding_model)
    elif args.model == "llm":
        scorer = LLMScorer(args.llm_model, prompt=args.llm_prompt, max_workers=args.llm_concurrency)
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
            args.use_weighted_emd,
            args.debug,
        )
    else:
        raise ValueError("Not implemented yet")

    scores: list[float] = []
    preds: list[float] = []

    data = process_data(df, args.label_prefix, args.score_method)
    for pair in tqdm(data):
        if args.aggregate_level == "sentence" or args.model == "nli" or bleu_topic_diagnostic:
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

    if hasattr(scorer, "print_filter_summary") and isinstance(scorer, EMDScorer):
        scorer.print_filter_summary()

    if not args.no_save_preds and not bleu_topic_diagnostic:
        if not os.path.exists("./results"):
            os.makedirs("./results", exist_ok=True)
        match args.aggregate_level:
            case "article":
                file_ = f"{args.language}_scores_article.csv"
            case "sentence":
                file_ = f"{args.language}_scores_sentence.csv"
            case _:
                raise ValueError(f"Invalid aggregate type: {args.aggregate_level}")
        pred_col = f"{args.model}_preds"
        pred_df = pl.from_dict(
            {
                "article": [pair.article for pair in data],
                "summary": [pair.summary for pair in data],
                "score": [pair.score for pair in data],
                pred_col: preds,
            }
        )
        if not os.path.exists(f"./results/{file_}"):
            df = pred_df
        else:
            existing_df = pl.read_csv(f"./results/{file_}")
            existing_pred_cols = [
                column
                for column in existing_df.columns
                if column not in {"article", "summary", "score", pred_col}
            ]
            df = pred_df.select(["article", "summary", "score"])
            if existing_pred_cols:
                df = df.join(
                    existing_df.select(["article", "summary", *existing_pred_cols]),
                    on=["article", "summary"],
                    how="left",
                )
            df = df.join(pred_df.select(["article", "summary", pred_col]), on=["article", "summary"], how="left")

        df.write_csv(f"./results/{file_}")


if __name__ == "__main__":
    main()
