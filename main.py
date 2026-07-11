import argparse
import os
import re
from pathlib import Path

import polars as pl

from scipy.stats import kendalltau, pearsonr, spearmanr
from tqdm.auto import tqdm

from src.models import BleuScorer, EmbeddingScorer, EMDScorer, LLMScorer, NLIScorer, RougeScorer, TfIdfScorer
from src.models.nli import NLI_SCORE_METHODS
from src.utils.data_utils import process_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Path to the CSV file containing the data.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory in which prediction CSVs are saved. Default: ./results.",
    )
    parser.add_argument(
        "--result-tag",
        type=str,
        default=None,
        help=(
            "Suffix used in prediction filenames. By default it is derived from the input CSV stem for "
            "non-canonical datasets; use an empty string to disable tagging."
        ),
    )
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
        "--nli-score-method",
        type=str,
        choices=NLI_SCORE_METHODS,
        default="preservation",
        help=(
            "Which NLI scoring method to use. shift is the previous signed mean stance shift. "
            "preservation is 1 - abs(E[Stance_summary] - E[Stance_source]) / 2."
        ),
    )
    parser.add_argument(
        "--emd-score-method",
        type=str,
        choices=["emd", "kl", "js", "argmax_ordinal", "argmax_exact", "euclidean", "itakura"],
        default="emd",
        help="Which score method to use for EMD baseline.",
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
        "--use-hebrew-morph-normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply lightweight Hebrew normalization and proclitic splitting before BLEU/ROUGE scoring.",
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
    parser.add_argument(
        "--soft-topic-token-jaccard",
        type=float,
        default=0.5,
        help="Token Jaccard threshold for soft topic filtering.",
    )
    parser.add_argument(
        "--soft-topic-char-jaccard",
        type=float,
        default=0.45,
        help="Char Jaccard threshold for soft topic filtering.",
    )
    parser.add_argument(
        "--soft-topic-fuzzy-ratio",
        type=float,
        default=0.82,
        help="Fuzzy ratio threshold for soft topic filtering.",
    )
    group.add_argument(
        "--use-embedding-topic-filtering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to filter pairs based on embedding similarity between topics.",
    )
    parser.add_argument(
        "--embedding-topic-threshold",
        "--embedding-topic-similarity-threshold",
        dest="embedding_topic_threshold",
        type=float,
        default=EMDScorer.DEFAULT_EMBEDDING_TOPIC_THRESHOLD,
        help="Topic embedding similarity threshold for embedding topic filtering.",
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
    parser.add_argument(
        "--divide-emd-by-sentence-count",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to divide the EMD distance by the number of sentences instead of kept pairs.",
    )
    parser.add_argument(
        "--penalize-filtered-emd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to count filtered EMD pairs as maximum divergence and divide by the number of sentences.",
    )
    parser.add_argument(
        "--use-gold-emd-topics",
        "--use-gold-topics",
        dest="use_gold_emd_topics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use dataset topic labels for EMD instead of generating topics with the topic model.",
    )
    args = parser.parse_args()
    if args.use_topic_filtering and args.use_topic_mismatch_filtering:
        parser.error("--use-topic-filtering and --use-topic-mismatch-filtering are mutually exclusive.")
    if args.use_topic_mismatch_filtering and args.model != "bleu":
        parser.error("--use-topic-mismatch-filtering is currently supported only with --model bleu.")
    if args.use_hebrew_morph_normalization and args.model not in {"bleu", "rouge1", "rouge2", "rougeL"}:
        parser.error("--use-hebrew-morph-normalization is supported only with BLEU and ROUGE models.")
    if args.use_hebrew_morph_normalization and args.language != "he":
        parser.error("--use-hebrew-morph-normalization is supported only with --language he.")
    if args.penalize_filtered_emd and args.emd_score_method in {"kl", "itakura"}:
        parser.error("--penalize-filtered-emd is not defined for --emd-score-method kl or itakura.")
    if args.use_gold_emd_topics and args.model != "emd":
        parser.error("--use-gold-emd-topics is supported only with --model emd.")
    return args


def prediction_column_name(args: argparse.Namespace) -> str:
    """Return a method-specific column name without conflating EMD variants."""
    if args.model == "nli":
        return f"nli_{args.nli_score_method}_preds"
    if args.model == "emd":
        return f"{args.emd_score_method}_preds"
    if args.use_hebrew_morph_normalization:
        return f"{args.model}_hebrew_morph_preds"
    return f"{args.model}_preds"


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
        scorer = BleuScorer(topic_filter=bleu_topic_filter, normalize_hebrew=args.use_hebrew_morph_normalization)
    elif args.model.startswith("rouge"):
        scorer = RougeScorer(args.model, normalize_hebrew=args.use_hebrew_morph_normalization)
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
            args.nli_score_method,
        )
    elif args.model == "emd":
        scorer = EMDScorer(
            matching_model_name=args.matching_model,
            topic_model_name=args.topic_model,
            stance_model_name=args.stance_model,
            aggregate=args.aggregate_level,
            language=args.language,
            entropy_threshold=args.entropy_threshold,
            use_topic_filtering=args.use_topic_filtering,
            use_soft_topic_filtering=args.use_soft_topic_filtering,
            soft_topic_token_jaccard=args.soft_topic_token_jaccard,
            soft_topic_char_jaccard=args.soft_topic_char_jaccard,
            soft_topic_fuzzy_ratio=args.soft_topic_fuzzy_ratio,
            use_embedding_topic_filtering=args.use_embedding_topic_filtering,
            embedding_topic_threshold=args.embedding_topic_threshold,
            use_dist_topic_score=args.use_dist_topic_score,
            use_weighted_emd=args.use_weighted_emd,
            debug=args.debug,
            score_method=args.emd_score_method,
            divide_by_sentence_count=args.divide_emd_by_sentence_count,
            penalize_filtered_pairs=args.penalize_filtered_emd,
            use_gold_topics=args.use_gold_emd_topics,
        )
    else:
        raise ValueError("Not implemented yet")

    scores: list[float] = []
    preds: list[float] = []

    data = process_data(df, args.label_prefix, args.score_method)
    for pair in tqdm(data):
        if (
            args.aggregate_level == "sentence"
            or args.model == "nli"
            or bleu_topic_diagnostic
            or (args.model == "emd" and args.use_gold_emd_topics)
        ):
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
        os.makedirs(args.results_dir, exist_ok=True)
        match args.aggregate_level:
            case "article":
                base_name = f"{args.language}_scores_article"
            case "sentence":
                base_name = f"{args.language}_scores_sentence"
            case _:
                raise ValueError(f"Invalid aggregate type: {args.aggregate_level}")
        canonical_stems = {"hebrew_normalized_majority", "english_normalized_majority"}
        input_stem = Path(args.input_file).stem
        result_tag = args.result_tag
        if result_tag is None:
            result_tag = "" if input_stem in canonical_stems else input_stem
        result_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", result_tag).strip("._-")
        file_ = f"{base_name}{f'__{result_tag}' if result_tag else ''}.csv"
        pred_col = prediction_column_name(args)
        pred_df = pl.from_dict(
            {
                "article": [pair.article for pair in data],
                "summary": [pair.summary for pair in data],
                "score": [pair.score for pair in data],
                pred_col: preds,
            }
        )
        output_path = os.path.join(args.results_dir, file_)
        if not os.path.exists(output_path):
            df = pred_df
        else:
            existing_df = pl.read_csv(output_path)
            existing_pred_cols = [
                column for column in existing_df.columns if column not in {"article", "summary", "score", pred_col}
            ]
            df = pred_df.select(["article", "summary", "score"])
            if existing_pred_cols:
                df = df.join(
                    existing_df.select(["article", "summary", *existing_pred_cols]),
                    on=["article", "summary"],
                    how="left",
                )
            df = df.join(pred_df.select(["article", "summary", pred_col]), on=["article", "summary"], how="left")

        df.write_csv(output_path)
        print(f"Saved predictions: {output_path}")


if __name__ == "__main__":
    main()
