#!/usr/bin/env python3
"""Build human-supported reference datasets and LLM-only adjudication queues.

The robust dataset retains an aligned source-summary pair only when each of the
four majority labels used by the reference score (source/summary topic and
stance) is supported by at least one of the two human annotators.  Excluded
rows are written to a queue for real manual adjudication; this script never
pretends that an automatic rule is a human adjudication.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HUMANS = ("annotator_A", "annotator_B")
LLMS = ("GPT", "Gemini")
TIE_BREAKER = "annotator_B"
SUFFIXES = (
    "article_topic",
    "article_stance",
    "summary_topic",
    "summary_stance",
)
STANCE_MAP = {
    "בעד": "Favor", "תומך": "Favor", "favor": "Favor",
    "against": "Against", "נגד": "Against", "נוגד": "Against", "<נגד>": "Against",
    "neutral": "Neutral", "נייטרלי": "Neutral", "ניטרלי": "Neutral",
    "עמדה נייטרלית": "Neutral", "<נייטרלי>": "Neutral",
}
DEFAULT_INPUTS = {
    "hebrew": ROOT / "data/datasets/hebrew_normalized_majority.csv",
    "english": ROOT / "data/datasets/english_normalized_majority.csv",
}


def missing(value: Any) -> bool:
    return pd.isna(value) or str(value).strip().lower() in {"", "nan", "none", "null"}


def normalize(value: Any, suffix: str) -> str | None:
    if missing(value):
        return None
    text = str(value).strip()
    if suffix.endswith("_topic"):
        return text
    bare = text.strip("<>[]{}()").strip()
    if not bare or bare.lower() in {"ריק", "העמדה", "blank", "empty"}:
        return None
    if bare in {"Favor", "Against", "Neutral"}:
        return bare
    return STANCE_MAP.get(text.lower()) or STANCE_MAP.get(text) or STANCE_MAP.get(bare) or text


def field_provenance(row: pd.Series, suffix: str) -> dict[str, Any]:
    majority = normalize(row.get(f"majority_{suffix}"), suffix)
    human_labels = [normalize(row.get(f"{p}_{suffix}"), suffix) for p in HUMANS]
    llm_labels = [normalize(row.get(f"{p}_{suffix}"), suffix) for p in LLMS]
    human_support = sum(label == majority for label in human_labels if label is not None)
    llm_support = sum(label == majority for label in llm_labels if label is not None)

    votes = [label for label in human_labels + llm_labels if label is not None]
    counts = Counter(votes)
    winners: set[str] = set()
    if counts:
        top = max(counts.values())
        winners = {label for label, count in counts.items() if count == top}
    tie_break_label = normalize(row.get(f"{TIE_BREAKER}_{suffix}"), suffix)
    human_tie_break = (
        len(winners) > 1 and tie_break_label in winners and majority == tie_break_label
    )

    if majority is None:
        category = "missing_majority"
    elif human_labels[0] == majority and human_labels[1] == majority:
        category = "both_humans_agree"
    elif human_tie_break and human_support > 0:
        category = "human_tie_break"
    elif human_support == 1:
        category = "one_human_supported_majority"
    elif human_support == 0 and llm_support > 0:
        category = "llm_only"
    elif human_support == 0:
        category = "unsupported"
    else:
        category = "human_supported_other"

    return {
        "category": category,
        "majority": majority,
        "human_support": human_support,
        "llm_support": llm_support,
    }


def document_count(df: pd.DataFrame) -> int:
    if not {"article", "summary"}.issubset(df.columns):
        return 0
    return int(df[["article", "summary"]].drop_duplicates().shape[0])


def process_language(language: str, input_path: Path, output_root: Path) -> dict[str, Any]:
    df = pd.read_csv(input_path)
    missing_columns = [
        f"{prefix}_{suffix}"
        for suffix in SUFFIXES
        for prefix in (*HUMANS, *LLMS, "majority")
        if f"{prefix}_{suffix}" not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"{input_path} is missing columns: {', '.join(missing_columns)}")

    field_counts = {suffix: Counter() for suffix in SUFFIXES}
    all_counts: Counter[str] = Counter()
    keep_mask: list[bool] = []
    queue_rows: list[dict[str, Any]] = []

    for source_index, row in df.iterrows():
        provenance = {suffix: field_provenance(row, suffix) for suffix in SUFFIXES}
        for suffix, detail in provenance.items():
            field_counts[suffix][detail["category"]] += 1
            all_counts[detail["category"]] += 1

        unsupported_fields = [
            suffix for suffix, detail in provenance.items() if detail["human_support"] == 0
        ]
        keep = not unsupported_fields
        keep_mask.append(keep)
        if not keep:
            queued = row.to_dict()
            queued["source_row_index"] = int(source_index)
            queued["fields_requiring_adjudication"] = ";".join(unsupported_fields)
            for suffix in SUFFIXES:
                queued[f"provenance_{suffix}"] = provenance[suffix]["category"]
                queued[f"adjudicated_{suffix}"] = ""
            queued["adjudicator_notes"] = ""
            queue_rows.append(queued)

    robust = df.loc[keep_mask].copy()
    dataset_dir = output_root / "data"
    queue_dir = output_root / "adjudication"
    report_dir = output_root / "reports"
    for directory in (dataset_dir, queue_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / f"{language}_human_supported_robustness.csv"
    queue_path = queue_dir / f"{language}_llm_only_adjudication_queue.csv"
    robust.to_csv(dataset_path, index=False)
    pd.DataFrame(queue_rows).to_csv(queue_path, index=False)

    original_docs = document_count(df)
    robust_docs = document_count(robust)
    report = {
        "language": language,
        "policy": "retain a pair iff all four majority reference labels have >=1 human supporter",
        "input": str(input_path.relative_to(ROOT) if input_path.is_relative_to(ROOT) else input_path),
        "output": str(dataset_path.relative_to(ROOT) if dataset_path.is_relative_to(ROOT) else dataset_path),
        "adjudication_queue": str(queue_path.relative_to(ROOT) if queue_path.is_relative_to(ROOT) else queue_path),
        "rows_original": len(df),
        "rows_retained": len(robust),
        "rows_excluded": len(df) - len(robust),
        "retained_percent": 100.0 * len(robust) / len(df) if len(df) else None,
        "documents_original": original_docs,
        "documents_retained": robust_docs,
        "documents_dropped": original_docs - robust_docs,
        "label_counts_all_fields": dict(all_counts),
        "label_counts_by_field": {k: dict(v) for k, v in field_counts.items()},
    }
    with (report_dir / f"{language}_reference_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def write_markdown(reports: list[dict[str, Any]], output_root: Path) -> Path:
    path = output_root / "reports/reference_robustness_summary.md"
    categories = (
        "both_humans_agree", "one_human_supported_majority", "human_tie_break",
        "llm_only", "unsupported", "missing_majority", "human_supported_other",
    )
    lines = [
        "# Reference-label robustness dataset", "",
        "The robustness dataset retains an aligned pair only when **all four** final reference labels "
        "(article topic/stance and summary topic/stance) are supported by at least one human. "
        "Excluded rows are placed in an adjudication queue; no automatic decision is described as manual adjudication.", "",
        "## Dataset retention", "",
        "| Language | Original pairs | Retained pairs | Excluded pairs | Retained documents | Dropped documents |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            f"| {report['language'].title()} | {report['rows_original']} | {report['rows_retained']} "
            f"({report['retained_percent']:.1f}%) | {report['rows_excluded']} | "
            f"{report['documents_retained']}/{report['documents_original']} | {report['documents_dropped']} |"
        )
    lines += ["", "## Mutually exclusive label provenance", "",
              "Counts pool the four reference fields; percentages are within language.", "",
              "| Language | Category | Count | Percent |", "|---|---|---:|---:|"]
    for report in reports:
        counts = report["label_counts_all_fields"]
        total = sum(counts.values())
        for category in categories:
            count = counts.get(category, 0)
            if count:
                lines.append(f"| {report['language'].title()} | `{category}` | {count} | {100*count/total:.1f}% |")
    lines += [
        "", "## Interpretation", "",
        "`human_tie_break` is reported separately from `one_human_supported_majority`, even though the "
        "tie-break label necessarily has one-human support. `llm_only` means the final label matches at least "
        "one LLM and neither human. `unsupported` means it matches none of the recorded voters.", "",
        "The filtered files are ready for a sensitivity rerun. The scientifically strongest final dataset is "
        "obtained by manually filling every `adjudicated_*` cell listed in each queue and then merging those "
        "decisions; until then, report this analysis as exclusion of labels without human support.", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["hebrew", "english", "all"], default="all")
    parser.add_argument("--input", type=Path, help="Custom input; requires a single --language.")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "rebuttal/reference_robustness",
        help="Root for datasets, queues, and reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input and args.language == "all":
        raise SystemExit("--input requires --language hebrew or --language english")
    languages = list(DEFAULT_INPUTS) if args.language == "all" else [args.language]
    reports = []
    for language in languages:
        input_path = args.input.resolve() if args.input else DEFAULT_INPUTS[language]
        report = process_language(language, input_path, args.output_root.resolve())
        reports.append(report)
        print(
            f"{language}: retained {report['rows_retained']}/{report['rows_original']} rows; "
            f"queue has {report['rows_excluded']} rows"
        )
    summary = write_markdown(reports, args.output_root.resolve())
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
