#!/usr/bin/env python3
"""Document-level bootstrap confidence intervals for the main correlation tables."""

import argparse
import csv

from collections import OrderedDict
from pathlib import Path

import numpy as np
import polars as pl

from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr

MODEL_COLUMNS = OrderedDict(
    [
        ("BLEU", "bleu_preds"),
        ("ROUGE-1", "rouge1_preds"),
        ("ROUGE-2", "rouge2_preds"),
        ("ROUGE-L", "rougeL_preds"),
        ("TF-IDF", "tf-idf_preds"),
        ("EMB", "emb_preds"),
        ("LLM", "llm_preds"),
        ("NLI model", "nli_preservation_preds"),
    ]
)
OURS_COLUMNS = OrderedDict([("Ours-EMD", "emd_preds"), ("Ours-JS", "js_preds")])
STRUCTURED_LLM_COLUMNS = OrderedDict(
    [
        ("Structured LLM EMD", "structured_llm_emd_preds"),
        ("Structured LLM Argmax", "structured_llm_argmax_preds"),
    ]
)
STATISTICS = OrderedDict(
    [
        ("Pearson", pearsonr),
        ("Spearman", spearmanr),
        ("Kendall", kendalltau),
    ]
)
LANGUAGES = OrderedDict([("he", "Hebrew"), ("en", "English")])
LEVELS = OrderedDict([("article", "Article-Level"), ("sentence", "Mean Sentence Article Score")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--baseline-sentence-dir",
        type=Path,
        default=Path("results/paper_baselines"),
        help="Directory containing the paper baseline sentence vectors.",
    )
    parser.add_argument("--output-tex", type=Path, default=Path("results/bootstrap_correlations.tex"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/bootstrap_correlations.csv"))
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be positive")
    return args


def correlation(function, predictions: np.ndarray, gold: np.ndarray) -> float:
    """Return a correlation statistic, or NaN for a degenerate resample."""
    if len(predictions) < 2 or np.ptp(predictions) == 0 or np.ptp(gold) == 0:
        return float("nan")
    return float(function(predictions, gold).statistic)


def column_pearson(predictions: np.ndarray, gold: np.ndarray) -> np.ndarray:
    centered_predictions = predictions - predictions.mean(axis=0)
    centered_gold = gold - gold.mean()
    numerator = np.sum(centered_predictions * centered_gold[:, np.newaxis], axis=0)
    denominator = np.sqrt(np.sum(centered_predictions**2, axis=0) * np.sum(centered_gold**2))
    return np.divide(
        numerator,
        denominator,
        out=np.full(predictions.shape[1], np.nan, dtype=float),
        where=denominator != 0,
    )


def column_kendall_tau_b(predictions: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Vectorized Kendall tau-b, including ties from repeated bootstrap documents."""
    first, second = np.triu_indices(len(gold), k=1)
    gold_differences = np.sign(gold[first] - gold[second])
    prediction_differences = np.sign(predictions[first] - predictions[second])
    numerator = np.sum(prediction_differences * gold_differences[:, np.newaxis], axis=0)
    gold_non_ties = np.count_nonzero(gold_differences)
    prediction_non_ties = np.count_nonzero(prediction_differences, axis=0)
    denominator = np.sqrt(prediction_non_ties * gold_non_ties)
    return np.divide(
        numerator,
        denominator,
        out=np.full(predictions.shape[1], np.nan, dtype=float),
        where=denominator != 0,
    )


def all_correlations(predictions: np.ndarray, gold: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Pearson": column_pearson(predictions, gold),
        "Spearman": column_pearson(rankdata(predictions, axis=0), rankdata(gold)),
        "Kendall": column_kendall_tau_b(predictions, gold),
    }


def document_groups(document_ids: list[str]) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for row_index, document_id in enumerate(document_ids):
        grouped.setdefault(document_id, []).append(row_index)
    return [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]


def bootstrap_indices(groups: list[np.ndarray], n_bootstrap: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Sample document clusters and retain every row belonging to each sampled document."""
    sampled = rng.integers(0, len(groups), size=(n_bootstrap, len(groups)))
    return [np.concatenate([groups[group_index] for group_index in draw]) for draw in sampled]


def analyze_file(
    input_path: Path,
    ours_sentence_path: Path,
    structured_llm_path: Path,
    language_code: str,
    level: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = pl.read_csv(input_path)
    required = {"article", "summary", "score", *MODEL_COLUMNS.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {', '.join(missing)}")

    model_columns = MODEL_COLUMNS.copy()
    ours_frame = pl.read_csv(ours_sentence_path)
    ours_required = {"article", "summary", *OURS_COLUMNS.values()}
    ours_missing = sorted(ours_required - set(ours_frame.columns))
    if ours_missing:
        raise ValueError(f"{ours_sentence_path} is missing required columns: {', '.join(ours_missing)}")

    emd_values = ours_frame[OURS_COLUMNS["Ours-EMD"]].to_numpy()
    js_values = ours_frame[OURS_COLUMNS["Ours-JS"]].to_numpy()
    if np.nanmin(emd_values) < 0.0 or np.nanmax(emd_values) > 2.0:
        raise ValueError(f"{ours_sentence_path} does not have the expected [0, 2] EMD score range")
    if np.nanmax(js_values) > 1e-12 or np.nanmin(js_values) < -2.0:
        raise ValueError(f"{ours_sentence_path} does not have the expected non-positive JS score range")

    joined_columns: dict[str, str] = {}
    ours_expressions: list[pl.Expr | str] = ["article", "summary"]
    for model, source_column in OURS_COLUMNS.items():
        joined_column = f"bootstrap_{source_column}"
        joined_columns[model] = joined_column
        ours_expressions.append(pl.col(source_column).alias(joined_column))
        model_columns[model] = joined_column
    ours_predictions = ours_frame.select(ours_expressions)
    frame = frame.join(ours_predictions, on=["article", "summary"], how="left", validate="1:1")
    if any(frame[column].null_count() for column in joined_columns.values()):
        raise ValueError(f"Could not match every row in {input_path} to {ours_sentence_path}")

    structured_frame = pl.read_csv(structured_llm_path)
    structured_required = {"article", "summary", *STRUCTURED_LLM_COLUMNS.values()}
    structured_missing = sorted(structured_required - set(structured_frame.columns))
    if structured_missing:
        raise ValueError(f"{structured_llm_path} is missing required columns: {', '.join(structured_missing)}")
    structured_emd = structured_frame[STRUCTURED_LLM_COLUMNS["Structured LLM EMD"]].to_numpy()
    structured_argmax = structured_frame[STRUCTURED_LLM_COLUMNS["Structured LLM Argmax"]].to_numpy()
    if np.nanmin(structured_emd) < 0.0 or np.nanmax(structured_emd) > 2.0:
        raise ValueError(f"{structured_llm_path} does not have the expected [0, 2] structured EMD score range")
    if np.nanmin(structured_argmax) < 0.0 or np.nanmax(structured_argmax) > 1.0:
        raise ValueError(f"{structured_llm_path} does not have the expected [0, 1] argmax score range")

    structured_joined_columns: dict[str, str] = {}
    structured_expressions: list[pl.Expr | str] = ["article", "summary"]
    for model, source_column in STRUCTURED_LLM_COLUMNS.items():
        joined_column = f"bootstrap_{source_column}"
        structured_joined_columns[model] = joined_column
        structured_expressions.append(pl.col(source_column).alias(joined_column))
        model_columns[model] = joined_column
    structured_predictions = structured_frame.select(structured_expressions)
    frame = frame.join(structured_predictions, on=["article", "summary"], how="left", validate="1:1")
    if any(frame[column].null_count() for column in structured_joined_columns.values()):
        raise ValueError(f"Could not match every row in {input_path} to {structured_llm_path}")

    gold = frame["score"].to_numpy()
    predictions = np.column_stack([frame[column].to_numpy() for column in model_columns.values()])
    groups = document_groups(frame["article"].to_list())
    language_index = list(LANGUAGES).index(language_code)
    # Reuse the same document resamples across aggregation levels. Besides being
    # paired, this makes a reused sentence-level Ours vector identical in both groups.
    rng = np.random.default_rng(np.random.SeedSequence([seed, language_index]))
    samples = bootstrap_indices(groups, n_bootstrap, rng)

    bootstrap_values = {statistic: np.empty((n_bootstrap, len(model_columns)), dtype=float) for statistic in STATISTICS}
    for bootstrap_index, row_indices in enumerate(samples):
        sample_correlations = all_correlations(predictions[row_indices], gold[row_indices])
        for statistic in STATISTICS:
            bootstrap_values[statistic][bootstrap_index] = sample_correlations[statistic]

    leave_one_document_out = {statistic: [] for statistic in STATISTICS}
    all_indices = np.arange(len(frame))
    for held_out_group in groups:
        row_indices = np.setdiff1d(all_indices, held_out_group, assume_unique=True)
        sample_correlations = all_correlations(predictions[row_indices], gold[row_indices])
        for statistic in STATISTICS:
            leave_one_document_out[statistic].append(sample_correlations[statistic])

    detailed_rows: list[dict[str, object]] = []
    intervals: dict[str, dict[str, tuple[float, float]]] = {statistic: {} for statistic in STATISTICS}
    estimates: dict[str, dict[str, float]] = {statistic: {} for statistic in STATISTICS}
    point_correlations = all_correlations(predictions, gold)
    for model_index, model in enumerate(model_columns):
        for statistic, function in STATISTICS.items():
            estimate = correlation(function, predictions[:, model_index], gold)
            if not np.isclose(estimate, point_correlations[statistic][model_index], atol=1e-12):
                raise RuntimeError(f"Vectorized {statistic} does not match SciPy for {input_path}, {model}")
            valid_bootstrap = bootstrap_values[statistic][:, model_index]
            valid_bootstrap = valid_bootstrap[np.isfinite(valid_bootstrap)]
            if len(valid_bootstrap) == 0:
                raise RuntimeError(f"All bootstrap replicates were degenerate for {input_path}, {model}, {statistic}")
            ci_low, ci_high = np.percentile(valid_bootstrap, [2.5, 97.5])
            loo = np.asarray(leave_one_document_out[statistic], dtype=float)[:, model_index]
            intervals[statistic][model] = (float(ci_low), float(ci_high))
            estimates[statistic][model] = estimate
            detailed_rows.append(
                {
                    "language": LANGUAGES[language_code],
                    "language_code": language_code,
                    "aggregate_level": level,
                    "model": model,
                    "statistic": statistic,
                    "estimate": estimate,
                    "ci_95_low": float(ci_low),
                    "ci_95_high": float(ci_high),
                    "bootstrap_median": float(np.median(valid_bootstrap)),
                    "probability_positive": float(np.mean(valid_bootstrap > 0)),
                    "loo_min": float(np.nanmin(loo)),
                    "loo_max": float(np.nanmax(loo)),
                    "loo_max_abs_change": float(np.nanmax(np.abs(loo - estimate))),
                    "valid_bootstrap_replicates": len(valid_bootstrap),
                    "n_rows": len(frame),
                    "n_documents": len(groups),
                    "source_file": (
                        str(ours_sentence_path)
                        if model in OURS_COLUMNS
                        else str(structured_llm_path)
                        if model in STRUCTURED_LLM_COLUMNS
                        else str(input_path)
                    ),
                    "seed": seed,
                }
            )

    stability_rows: list[dict[str, object]] = []
    models = list(model_columns)
    for statistic in STATISTICS:
        point_leader = max(models, key=lambda model: estimates[statistic][model])
        replicate_matrix = bootstrap_values[statistic]
        finite_rows = np.all(np.isfinite(replicate_matrix), axis=1)
        replicate_leaders = np.argmax(replicate_matrix[finite_rows], axis=1)
        leader_rates = np.bincount(replicate_leaders, minlength=len(models)) / len(replicate_leaders)
        most_frequent_index = int(np.argmax(leader_rates))
        leader_low, leader_high = intervals[statistic][point_leader]
        overlapping = [
            model
            for model in models
            if intervals[statistic][model][0] <= leader_high and intervals[statistic][model][1] >= leader_low
        ]
        point_details = next(
            row for row in detailed_rows if row["model"] == point_leader and row["statistic"] == statistic
        )
        stability_rows.append(
            {
                "language": LANGUAGES[language_code],
                "aggregate_level": level,
                "statistic": statistic,
                "point_leader": point_leader,
                "point_estimate": estimates[statistic][point_leader],
                "point_leader_first_probability": float(leader_rates[models.index(point_leader)]),
                "most_frequent_bootstrap_leader": models[most_frequent_index],
                "most_frequent_leader_probability": float(leader_rates[most_frequent_index]),
                "point_leader_loo_max_abs_change": point_details["loo_max_abs_change"],
                "overlapping_intervals": ", ".join(overlapping),
            }
        )

    return detailed_rows, stability_rows


def format_interval(row: dict[str, object]) -> str:
    return f"{row['estimate']:.3f} [{row['ci_95_low']:.3f}, {row['ci_95_high']:.3f}]"


def latex_escape(value: str) -> str:
    return value.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def render_correlation_table(
    language_code: str,
    rows: list[dict[str, object]],
    n_bootstrap: int,
) -> str:
    lookup = {
        (str(row["aggregate_level"]), str(row["model"]), str(row["statistic"])): row
        for row in rows
        if row["language_code"] == language_code
    }
    n_documents = int(next(iter(lookup.values()))["n_documents"])
    models = list(
        dict.fromkeys(
            str(row["model"])
            for row in rows
            if row["language_code"] == language_code and row["aggregate_level"] == "article"
        )
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{l|ccc|ccc}",
        r"\hline",
        r" & \multicolumn{3}{c|}{Article-Level} & \multicolumn{3}{c}{Mean Sentence Article Score} \\",
        r"Model & Pearson & Spearman & Kendall & Pearson & Spearman & Kendall \\",
        r"\hline",
    ]
    for model in models:
        cells = [format_interval(lookup[(level, model, statistic)]) for level in LEVELS for statistic in STATISTICS]
        lines.append(f"{latex_escape(model)} & " + " & ".join(cells) + r" \\")
    language_name = LANGUAGES[language_code]
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            (
                rf"\caption{{\textbf{{{language_name} document-bootstrap results.}} Point correlations and "
                rf"percentile 95\% confidence intervals from {n_bootstrap:,} clustered bootstrap resamples of "
                rf"{n_documents} documents. All rows associated with a sampled document are resampled together. "
                r"The rerun sentence-aggregated Ours and structured LLM vectors are reused for both column groups. "
                r"Intervals are uncertainty estimates, not simultaneous confidence bands. Because the intervals "
                r"overlap extensively, these results should not be used to assert a precise ranking among metrics.}"
            ),
            rf"\label{{tab:{language_code}_bootstrap_correlations}}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def render_stability_table(rows: list[dict[str, object]], n_bootstrap: int) -> str:
    statistic_symbols = {"Pearson": r"$r$", "Spearman": r"$\rho$", "Kendall": r"$\tau$"}
    level_names = {"article": "Article", "sentence": "Mean sentence"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lll|p{0.15\textwidth}p{0.15\textwidth}c|p{0.27\textwidth}}",
        r"\hline",
        (
            r"Language & Level & Corr. & Largest point estimate / $P(\mathrm{largest})$ & "
            r"Most often largest / $P(\mathrm{largest})$ & Max LODO $|\Delta|$ & Overlapping 95\% CIs \\"
        ),
        r"\hline",
    ]
    for row in rows:
        overlap = latex_escape(str(row["overlapping_intervals"]))
        lines.append(
            f"{row['language']} & {level_names[str(row['aggregate_level'])]} & "
            f"{statistic_symbols[str(row['statistic'])]} & "
            f"{latex_escape(str(row['point_leader']))} / {100 * float(row['point_leader_first_probability']):.1f}\\% & "
            f"{latex_escape(str(row['most_frequent_bootstrap_leader']))} / "
            f"{100 * float(row['most_frequent_leader_probability']):.1f}\\% & "
            f"{float(row['point_leader_loo_max_abs_change']):.3f} & {overlap} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            (
                rf"\caption{{\textbf{{Correlation stability and sensitivity.}} $P(\mathrm{{largest}})$ is the "
                rf"fraction of {n_bootstrap:,} clustered document-bootstrap samples in which a metric had the "
                r"largest correlation. LODO removes one entire document at a time; Max LODO $|\Delta|$ is the "
                r"largest absolute change for the metric with the largest full-data estimate. The final column "
                r"lists every metric whose marginal bootstrap interval overlaps that metric's interval. These "
                r"descriptive maxima are not ranking claims: high overlap and low selection probabilities indicate "
                r"that precise metric ordering is not stable.}"
            ),
            r"\label{tab:bootstrap_stability}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    detailed_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    for language_code in LANGUAGES:
        for level in LEVELS:
            baseline_dir = args.results_dir if level == "article" else args.baseline_sentence_dir
            input_path = baseline_dir / f"{language_code}_scores_{level}.csv"
            ours_sentence_path = args.results_dir / f"{language_code}_scores_sentence.csv"
            structured_llm_path = args.results_dir / f"{language_code}_structured_llm_baseline.csv"
            file_rows, file_stability = analyze_file(
                input_path=input_path,
                ours_sentence_path=ours_sentence_path,
                structured_llm_path=structured_llm_path,
                language_code=language_code,
                level=level,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            detailed_rows.extend(file_rows)
            stability_rows.extend(file_stability)

    write_csv(args.output_csv, detailed_rows)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    latex = [
        "% Generated by scripts/bootstrap_correlations.py; do not edit by hand.",
        f"% Seed: {args.seed}; clustered document bootstrap replicates: {args.n_bootstrap}",
        "% Rerun Ours and structured LLM vectors are reused for both aggregation groups.",
        render_correlation_table("he", detailed_rows, args.n_bootstrap),
        render_correlation_table("en", detailed_rows, args.n_bootstrap),
        render_stability_table(stability_rows, args.n_bootstrap),
        "",
    ]
    args.output_tex.write_text("\n\n".join(latex), encoding="utf-8")
    print(f"Wrote {args.output_tex}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
