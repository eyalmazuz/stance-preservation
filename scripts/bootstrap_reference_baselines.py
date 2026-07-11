#!/usr/bin/env python3
"""Bootstrap baseline correlations for the human-supported robustness rerun."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr


MODELS = OrderedDict([
    ("BLEU", "bleu_preds"), ("ROUGE-1", "rouge1_preds"),
    ("ROUGE-2", "rouge2_preds"), ("ROUGE-L", "rougeL_preds"),
    ("TF-IDF", "tf-idf_preds"), ("EMB", "emb_preds"), ("LLM", "llm_preds"),
    ("Structured LLM EMD", "structured_llm_emd_preds"),
    ("Structured LLM Argmax", "structured_llm_argmax_preds"),
])
STATS = OrderedDict([("Pearson", pearsonr), ("Spearman", spearmanr), ("Kendall", kendalltau)])
FILES = {
    ("Hebrew", "article"): "he_scores_article__hebrew_human_supported_robustness.csv",
    ("Hebrew", "sentence"): "he_scores_sentence__hebrew_human_supported_robustness.csv",
    ("English", "article"): "en_scores_article__english_human_supported_robustness.csv",
    ("English", "sentence"): "en_scores_sentence__english_human_supported_robustness.csv",
}
STRUCTURED_FILES = {
    "Hebrew": "he_structured_llm_baseline__hebrew_human_supported_robustness.csv",
    "English": "en_structured_llm_baseline__english_human_supported_robustness.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/reference_robustness"))
    parser.add_argument("--output-dir", type=Path, default=Path("rebuttal/reference_robustness/results"))
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def corr(function, prediction: np.ndarray, gold: np.ndarray) -> float:
    if len(gold) < 2 or np.ptp(prediction) == 0 or np.ptp(gold) == 0:
        return float("nan")
    return float(function(prediction, gold).statistic)


def column_pearson(predictions: np.ndarray, gold: np.ndarray) -> np.ndarray:
    x = predictions - predictions.mean(axis=0)
    y = gold - gold.mean()
    denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y))
    return np.divide(np.sum(x * y[:, None], axis=0), denominator,
                     out=np.full(predictions.shape[1], np.nan), where=denominator != 0)


def column_kendall(predictions: np.ndarray, gold: np.ndarray) -> np.ndarray:
    first, second = np.triu_indices(len(gold), k=1)
    gold_diff = np.sign(gold[first] - gold[second])
    pred_diff = np.sign(predictions[first] - predictions[second])
    denominator = np.sqrt(np.count_nonzero(pred_diff, axis=0) * np.count_nonzero(gold_diff))
    return np.divide(np.sum(pred_diff * gold_diff[:, None], axis=0), denominator,
                     out=np.full(predictions.shape[1], np.nan), where=denominator != 0)


def all_correlations(predictions: np.ndarray, gold: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Pearson": column_pearson(predictions, gold),
        "Spearman": column_pearson(rankdata(predictions, axis=0), rankdata(gold)),
        "Kendall": column_kendall(predictions, gold),
    }


def analyze(path: Path, language: str, level: str, n_bootstrap: int, seed: int) -> list[dict[str, object]]:
    frame = pl.read_csv(path)
    structured_path = path.parent / STRUCTURED_FILES[language]
    structured = pl.read_csv(structured_path).select([
        "article", "summary", "structured_llm_emd_preds", "structured_llm_argmax_preds",
    ])
    frame = frame.join(structured, on=["article", "summary"], how="left", validate="1:1")
    required = {"article", "summary", "score", *MODELS.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks completed baseline columns: {', '.join(missing)}")
    if frame.select(pl.struct(["article", "summary"]).n_unique()).item() != len(frame):
        raise ValueError(f"{path} does not contain one row per document-summary pair")

    gold = frame["score"].to_numpy()
    prediction_matrix = np.column_stack([frame[column].to_numpy() for column in MODELS.values()])
    language_seed = 0 if language == "Hebrew" else 1
    level_seed = 0 if level == "article" else 1
    rng = np.random.default_rng(np.random.SeedSequence([seed, language_seed, level_seed]))
    samples = rng.integers(0, len(frame), size=(n_bootstrap, len(frame)))
    all_indices = np.arange(len(frame))
    boot = {stat: np.empty((n_bootstrap, len(MODELS))) for stat in STATS}
    for sample_index, indices in enumerate(samples):
        values = all_correlations(prediction_matrix[indices], gold[indices])
        for statistic in STATS:
            boot[statistic][sample_index] = values[statistic]
    loo = {stat: [] for stat in STATS}
    for held_out in all_indices:
        values = all_correlations(prediction_matrix[all_indices != held_out], gold[all_indices != held_out])
        for statistic in STATS:
            loo[statistic].append(values[statistic])
    rows: list[dict[str, object]] = []
    for model_index, (model, column) in enumerate(MODELS.items()):
        prediction = frame[column].to_numpy()
        for statistic, function in STATS.items():
            estimate = corr(function, prediction, gold)
            bootstrap = boot[statistic][:, model_index]
            bootstrap = bootstrap[np.isfinite(bootstrap)]
            loo_values = np.asarray(loo[statistic])[:, model_index]
            low, high = np.percentile(bootstrap, [2.5, 97.5])
            rows.append({
                "language": language, "aggregate_level": level, "model": model, "statistic": statistic,
                "estimate": estimate, "ci_95_low": float(low), "ci_95_high": float(high),
                "bootstrap_median": float(np.median(bootstrap)),
                "probability_positive": float(np.mean(bootstrap > 0)),
                "loo_min": float(np.nanmin(loo_values)), "loo_max": float(np.nanmax(loo_values)),
                "loo_max_abs_change": float(np.nanmax(np.abs(loo_values - estimate))),
                "valid_bootstrap_replicates": len(bootstrap), "n_documents": len(frame),
                "source_file": str(path), "seed": seed,
            })
    return rows


def markdown(rows: list[dict[str, object]], n_bootstrap: int) -> str:
    lookup = {(r["language"], r["aggregate_level"], r["model"], r["statistic"]): r for r in rows}
    lines = [
        "# Human-supported reference robustness: baseline bootstrap", "",
        f"Percentile 95% confidence intervals from {n_bootstrap:,} document-level bootstrap resamples. "
        "NLI and the proposed EMD/JS methods are intentionally omitted. Article-level EMB/LLM predictions are reused from the "
        "original run because filtering sentence annotations cannot change full-document predictions; their "
        "correlations are recomputed against the filtered reference scores. The structured LLM produces one "
        "document-level vector, reported in both column groups for comparison with the paper table.", "",
    ]
    for language in ("Hebrew", "English"):
        lines += [f"## {language}", "",
                  "| Metric | Article Pearson | Article Spearman | Article Kendall | Sentence Pearson | Sentence Spearman | Sentence Kendall |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for model in MODELS:
            cells=[]
            for level in ("article", "sentence"):
                for statistic in STATS:
                    row=lookup[(language,level,model,statistic)]
                    cells.append(f"{row['estimate']:.3f} [{row['ci_95_low']:.3f}, {row['ci_95_high']:.3f}]")
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        lines += ["", "Confidence intervals overlap extensively; these estimates should not be used to assert a precise metric ranking.", ""]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows=[]
    for (language, level), filename in FILES.items():
        rows.extend(analyze(args.results_dir / filename, language, level, args.n_bootstrap, args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path=args.output_dir / "baseline_bootstrap.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    md_path=args.output_dir / "baseline_bootstrap.md"
    md_path.write_text(markdown(rows,args.n_bootstrap),encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
