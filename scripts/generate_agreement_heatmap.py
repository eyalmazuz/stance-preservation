from __future__ import annotations

import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT_DIR / "data" / "datasets" / "hebrew_data_labeled.csv"
DEFAULT_OUTPUT = ROOT_DIR / "heatmap.png"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 2x2 inter-annotator agreement heatmap grid using Cohen's kappa."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--annotators",
        nargs="*",
        type=str,
        help="Optional list of annotators to include in the heatmap. If not provided, all annotators are included.",
    )
    return parser.parse_args()


def cohen_kappa(left: list[str | None], right: list[str | None]) -> float:
    paired = [(lval, rval) for lval, rval in zip(left, right, strict=True) if lval is not None and rval is not None]
    if not paired:
        return float("nan")

    left_values = [lval for lval, _ in paired]
    right_values = [rval for _, rval in paired]
    labels = sorted(set(left_values) | set(right_values))
    total = len(paired)

    agreement = sum(1 for lval, rval in paired if lval == rval) / total
    expected = 0.0
    for label in labels:
        left_prob = left_values.count(label) / total
        right_prob = right_values.count(label) / total
        expected += left_prob * right_prob

    if expected == 1.0:
        return 1.0 if agreement == 1.0 else 0.0

    return (agreement - expected) / (1 - expected)


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return str(value).strip().lower() in {"", "nan", "null", "none"}


def normalize_topic(value: object) -> str | None:
    if is_missing(value):
        return None
    return str(value).strip()


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


def normalize_value(value: object, suffix: str) -> str | None:
    if suffix.endswith("_topic"):
        return normalize_topic(value)
    return normalize_stance(value)


def find_label_columns(columns: list[str], suffix: str) -> list[str]:
    return [column for column in columns if column.endswith(f"_{suffix}")]


def order_label_columns(columns: list[str], suffix: str) -> list[str]:
    majority_column = f"majority_{suffix}"
    non_majority_columns = [column for column in columns if column != majority_column]
    if majority_column in columns:
        return [*non_majority_columns, majority_column]
    return non_majority_columns


def display_name(column: str, suffix: str) -> str:
    if column == f"majority_{suffix}":
        return "Majority"
    prefix = column[: -len(f"_{suffix}")]
    return prefix.rstrip("_")


def collect_normalized_columns(
    df: pl.DataFrame,
    suffix: str,
    annotators: list[str] | None = None,
) -> tuple[list[str], dict[str, list[str | None]]]:
    raw_columns = find_label_columns(df.columns, suffix)
    
    if annotators:
        filtered_columns = []
        for col in raw_columns:
            if col == f"majority_{suffix}":
                filtered_columns.append(col)
                continue
            
            prefix = col[: -len(f"_{suffix}")]
            if prefix in annotators:
                filtered_columns.append(col)
    else:
        filtered_columns = raw_columns

    columns = order_label_columns(filtered_columns, suffix)
    values: dict[str, list[str | None]] = {}

    for column in columns:
        series_values = df.get_column(column).to_list()
        values[column] = [normalize_value(value, suffix) for value in series_values]

    return columns, values


def build_kappa_matrix(
    ordered_columns: list[str],
    normalized_values: dict[str, list[str | None]],
) -> np.ndarray:
    size = len(ordered_columns)
    matrix = np.empty((size, size), dtype=float)

    for row_index, left_column in enumerate(ordered_columns):
        for col_index, right_column in enumerate(ordered_columns):
            matrix[row_index, col_index] = cohen_kappa(
                normalized_values[left_column],
                normalized_values[right_column],
            )

    return matrix


def suffix_title(suffix: str) -> str:
    area, label_type = suffix.split("_", 1)
    return f"{area.capitalize()} {label_type.capitalize()}"


def plot_heatmaps(df: pl.DataFrame, output_png: Path, annotators: list[str] | None = None) -> None:
    sns.set_theme(style="white", font_scale=0.9)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes_by_suffix = {
        "summary_topic": axes[0, 0],
        "summary_stance": axes[0, 1],
        "article_topic": axes[1, 0],
        "article_stance": axes[1, 1],
    }

    for suffix, axis in axes_by_suffix.items():
        ordered_columns, normalized_values = collect_normalized_columns(df, suffix, annotators)
        if not ordered_columns:
            raise ValueError(f"No columns found for suffix: {suffix}")
        matrix = build_kappa_matrix(ordered_columns, normalized_values)
        labels = [display_name(column, suffix) for column in ordered_columns]
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        sns.heatmap(
            matrix,
            ax=axis,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            xticklabels=labels,
            yticklabels=labels,
            cbar=axis is axes[0, 1],
            cbar_kws={"shrink": 0.85, "label": "Cohen's kappa"},
        )
        axis.set_title(suffix_title(suffix))
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(axis="y", rotation=0)

    fig.suptitle("Inter-Annotator Agreement", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    
    annotators = None
    if args.annotators:
        annotators = []
        for a in args.annotators:
            annotators.extend([x.strip() for x in a.split(",") if x.strip()])
            
    df = pl.read_csv(args.input_csv)
    plot_heatmaps(df, args.output_png, annotators)
    print(f"Read {len(df)} rows from {args.input_csv}")
    print(f"Wrote heatmap to {args.output_png}")


if __name__ == "__main__":
    main()
