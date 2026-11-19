# /// script
# dependencies = [
#   "polars",
# ]
# ///

from __future__ import annotations

import argparse
import json
import math

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

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

DEFAULT_HUMAN_PREFIXES = ("annotator_A", "annotator_B")
DEFAULT_LLM_PREFIXES = ("GPT", "Gemini")
DEFAULT_TIE_BREAK_PREFIX = "annotator_B"
SIDE_TO_SUFFIXES = {
    "article": ("article_topic", "article_stance"),
    "summary": ("summary_topic", "summary_stance"),
}


@dataclass
class ProvenanceStats:
    suffix: str
    total_majority_labels: int = 0
    human_only: int = 0
    llm_only: int = 0
    mixed_human_llm: int = 0
    unsupported: int = 0
    any_human_support: int = 0
    any_llm_support: int = 0
    human_support_advantage: int = 0
    llm_support_advantage: int = 0
    equal_human_llm_support: int = 0
    top_vote_ties: int = 0
    ties_resolved_by_tie_breaker: int = 0
    ties_not_resolved_by_tie_breaker: int = 0
    majority_matches_tie_breaker: int = 0


@dataclass
class AgreementStats:
    suffix: str
    prefixes: list[str]
    columns: list[str]
    items_total: int
    items_with_at_least_two_labels: int
    labels_total: int
    categories: int
    observed_agreement: float | None
    krippendorff_alpha_nominal: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a normalized majority CSV: report whether final majority labels are supported by human "
            "annotators, LLM annotators, or both; count human tie-breaker resolutions; and compute "
            "human-only Krippendorff's alpha."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Path to a normalized majority CSV.")
    parser.add_argument(
        "--side",
        choices=["article", "summary", "both"],
        default="article",
        help="Which labels to audit. article is the reference/source side. Default: article.",
    )
    parser.add_argument(
        "--human-prefixes",
        default=",".join(DEFAULT_HUMAN_PREFIXES),
        help="Comma-separated human annotator prefixes. Default: annotator_A,annotator_B.",
    )
    parser.add_argument(
        "--llm-prefixes",
        default=",".join(DEFAULT_LLM_PREFIXES),
        help="Comma-separated LLM annotator prefixes. Default: GPT,Gemini.",
    )
    parser.add_argument(
        "--vote-prefixes",
        default=None,
        help=(
            "Comma-separated prefixes used to reconstruct vote ties. "
            "Default: all present human and LLM prefixes."
        ),
    )
    parser.add_argument(
        "--tie-break-prefix",
        default=DEFAULT_TIE_BREAK_PREFIX,
        help="Human prefix used as the tie-breaker when the majority CSV was created. Default: annotator_B.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write the same report as JSON.",
    )
    return parser.parse_args()


def parse_prefixes(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "null", "none"}


def normalize_stance(value: object) -> str | None:
    if is_missing(value):
        return None
    cleaned = str(value).strip()
    cleaned_no_brackets = cleaned.strip("<>[]{}()").strip()
    if not cleaned_no_brackets or cleaned_no_brackets in {"ריק", "העמדה", "blank", "empty"}:
        return None
    if cleaned in {"Favor", "Against", "Neutral"}:
        return cleaned
    return STANCE_MAP.get(cleaned.lower()) or STANCE_MAP.get(cleaned) or STANCE_MAP.get(cleaned_no_brackets) or cleaned


def normalize_label(value: object, suffix: str) -> str | None:
    if suffix.endswith("_stance"):
        return normalize_stance(value)
    if is_missing(value):
        return None
    return str(value).strip()


def selected_suffixes(side: str) -> list[str]:
    if side == "both":
        return [*SIDE_TO_SUFFIXES["article"], *SIDE_TO_SUFFIXES["summary"]]
    return list(SIDE_TO_SUFFIXES[side])


def source_column(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def present_prefixes(df: pl.DataFrame, prefixes: list[str], suffix: str) -> list[str]:
    return [prefix for prefix in prefixes if source_column(prefix, suffix) in df.columns]


def support_counts(
    row: dict[str, Any],
    suffix: str,
    prefixes: list[str],
    majority_label: str,
) -> int:
    count = 0
    for prefix in prefixes:
        label = normalize_label(row.get(source_column(prefix, suffix)), suffix)
        if label == majority_label:
            count += 1
    return count


def update_support_stats(
    stats: ProvenanceStats,
    human_support: int,
    llm_support: int,
) -> None:
    if human_support > 0:
        stats.any_human_support += 1
    if llm_support > 0:
        stats.any_llm_support += 1

    if human_support > 0 and llm_support > 0:
        stats.mixed_human_llm += 1
    elif human_support > 0:
        stats.human_only += 1
    elif llm_support > 0:
        stats.llm_only += 1
    else:
        stats.unsupported += 1

    if human_support > llm_support:
        stats.human_support_advantage += 1
    elif llm_support > human_support:
        stats.llm_support_advantage += 1
    elif human_support > 0:
        stats.equal_human_llm_support += 1


def update_tie_stats(
    stats: ProvenanceStats,
    row: dict[str, Any],
    suffix: str,
    vote_prefixes: list[str],
    tie_break_prefix: str,
    majority_label: str,
) -> None:
    vote_labels = []
    tie_break_label = None
    for prefix in vote_prefixes:
        label = normalize_label(row.get(source_column(prefix, suffix)), suffix)
        if label is None:
            continue
        vote_labels.append(label)
        if prefix == tie_break_prefix:
            tie_break_label = label

    if tie_break_label == majority_label:
        stats.majority_matches_tie_breaker += 1

    if not vote_labels:
        return

    counts = Counter(vote_labels)
    max_votes = max(counts.values())
    winners = {label for label, count in counts.items() if count == max_votes}
    if len(winners) <= 1:
        return

    stats.top_vote_ties += 1
    if tie_break_label in winners and majority_label == tie_break_label:
        stats.ties_resolved_by_tie_breaker += 1
    else:
        stats.ties_not_resolved_by_tie_breaker += 1


def analyze_provenance(
    df: pl.DataFrame,
    suffix: str,
    human_prefixes: list[str],
    llm_prefixes: list[str],
    vote_prefixes: list[str],
    tie_break_prefix: str,
) -> ProvenanceStats:
    stats = ProvenanceStats(suffix=suffix)
    majority_column = f"majority_{suffix}"
    if majority_column not in df.columns:
        raise ValueError(f"Missing majority column: {majority_column}")

    present_humans = present_prefixes(df, human_prefixes, suffix)
    present_llms = present_prefixes(df, llm_prefixes, suffix)
    present_voters = present_prefixes(df, vote_prefixes, suffix)

    for row in df.iter_rows(named=True):
        majority_label = normalize_label(row.get(majority_column), suffix)
        if majority_label is None:
            continue

        stats.total_majority_labels += 1
        human_support = support_counts(row, suffix, present_humans, majority_label)
        llm_support = support_counts(row, suffix, present_llms, majority_label)
        update_support_stats(stats, human_support, llm_support)
        update_tie_stats(stats, row, suffix, present_voters, tie_break_prefix, majority_label)

    return stats


def krippendorff_alpha_nominal(items: list[list[str | None]]) -> tuple[float | None, float | None, int, int, int]:
    category_counts: Counter[str] = Counter()
    observed_disagreement_num = 0
    observed_disagreement_den = 0
    items_with_at_least_two_labels = 0

    for labels in items:
        valid_labels = [label for label in labels if label is not None]
        category_counts.update(valid_labels)
        n_labels = len(valid_labels)
        if n_labels < 2:
            continue

        items_with_at_least_two_labels += 1
        counts = Counter(valid_labels)
        observed_disagreement_num += sum(count * (n_labels - count) for count in counts.values())
        observed_disagreement_den += n_labels * (n_labels - 1)

    labels_total = sum(category_counts.values())
    if observed_disagreement_den == 0 or labels_total < 2:
        return None, None, items_with_at_least_two_labels, labels_total, len(category_counts)

    observed_disagreement = observed_disagreement_num / observed_disagreement_den
    expected_disagreement = sum(
        count * (labels_total - count)
        for count in category_counts.values()
    ) / (labels_total * (labels_total - 1))

    observed_agreement = 1.0 - observed_disagreement
    if expected_disagreement == 0:
        alpha = 1.0 if observed_disagreement == 0 else None
    else:
        alpha = 1.0 - observed_disagreement / expected_disagreement

    return observed_agreement, alpha, items_with_at_least_two_labels, labels_total, len(category_counts)


def analyze_human_agreement(
    df: pl.DataFrame,
    suffix: str,
    human_prefixes: list[str],
) -> AgreementStats:
    prefixes = present_prefixes(df, human_prefixes, suffix)
    columns = [source_column(prefix, suffix) for prefix in prefixes]
    items = []
    for row in df.iter_rows(named=True):
        items.append([normalize_label(row.get(column), suffix) for column in columns])

    observed_agreement, alpha, items_with_two, labels_total, category_count = krippendorff_alpha_nominal(items)
    return AgreementStats(
        suffix=suffix,
        prefixes=prefixes,
        columns=columns,
        items_total=len(items),
        items_with_at_least_two_labels=items_with_two,
        labels_total=labels_total,
        categories=category_count,
        observed_agreement=observed_agreement,
        krippendorff_alpha_nominal=alpha,
    )


def pct(count: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{count / total:.1%}"


def fmt_float(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:.3f}"


def sum_provenance_stats(stats_list: list[ProvenanceStats]) -> ProvenanceStats:
    total = ProvenanceStats(suffix="ALL_SELECTED")
    for stats in stats_list:
        for field_name, value in asdict(stats).items():
            if field_name == "suffix":
                continue
            setattr(total, field_name, getattr(total, field_name) + value)
    return total


def print_provenance(stats: ProvenanceStats) -> None:
    total = stats.total_majority_labels
    print(f"\nFinal majority support: {stats.suffix}")
    print(f"  labels with final majority: {total}")
    print(f"  any human support:          {stats.any_human_support:>5} ({pct(stats.any_human_support, total)})")
    print(f"  any LLM support:            {stats.any_llm_support:>5} ({pct(stats.any_llm_support, total)})")
    print(f"  human-only support:         {stats.human_only:>5} ({pct(stats.human_only, total)})")
    print(f"  LLM-only support:           {stats.llm_only:>5} ({pct(stats.llm_only, total)})")
    print(f"  mixed human+LLM support:    {stats.mixed_human_llm:>5} ({pct(stats.mixed_human_llm, total)})")
    print(f"  unsupported by these groups:{stats.unsupported:>5} ({pct(stats.unsupported, total)})")
    print(
        f"  human support > LLM support:{stats.human_support_advantage:>5} "
        f"({pct(stats.human_support_advantage, total)})"
    )
    print(f"  LLM support > human support:{stats.llm_support_advantage:>5} ({pct(stats.llm_support_advantage, total)})")
    print(
        f"  equal nonzero support:      {stats.equal_human_llm_support:>5} "
        f"({pct(stats.equal_human_llm_support, total)})"
    )
    print(f"  top-vote ties:              {stats.top_vote_ties:>5} ({pct(stats.top_vote_ties, total)})")
    print(
        "  ties resolved by tie-breaker:"
        f"{stats.ties_resolved_by_tie_breaker:>4} "
        f"({pct(stats.ties_resolved_by_tie_breaker, stats.top_vote_ties)} of ties; "
        f"{pct(stats.ties_resolved_by_tie_breaker, total)} of labels)"
    )
    print(
        "  ties not resolved by tie-breaker:"
        f"{stats.ties_not_resolved_by_tie_breaker:>2} "
        f"({pct(stats.ties_not_resolved_by_tie_breaker, stats.top_vote_ties)} of ties)"
    )
    print(
        "  final majority matches tie-breaker:"
        f"{stats.majority_matches_tie_breaker:>3} ({pct(stats.majority_matches_tie_breaker, total)})"
    )


def print_agreement(stats: AgreementStats) -> None:
    print(f"\nHuman-only agreement: {stats.suffix}")
    print(f"  human prefixes:        {', '.join(stats.prefixes) if stats.prefixes else 'none'}")
    print(f"  columns:               {', '.join(stats.columns) if stats.columns else 'none'}")
    print(f"  items total:           {stats.items_total}")
    print(f"  items with >=2 labels: {stats.items_with_at_least_two_labels}")
    print(f"  labels total:          {stats.labels_total}")
    print(f"  categories:            {stats.categories}")
    print(f"  observed agreement:    {fmt_float(stats.observed_agreement)}")
    print(f"  Krippendorff alpha:    {fmt_float(stats.krippendorff_alpha_nominal)}")


def main() -> None:
    args = parse_args()
    df = pl.read_csv(args.input_csv)
    suffixes = selected_suffixes(args.side)
    human_prefixes = parse_prefixes(args.human_prefixes)
    llm_prefixes = parse_prefixes(args.llm_prefixes)
    vote_prefixes = parse_prefixes(args.vote_prefixes) or [
        prefix
        for prefix in [*human_prefixes, *llm_prefixes]
        if any(source_column(prefix, suffix) in df.columns for suffix in suffixes)
    ]

    provenance_stats = [
        analyze_provenance(df, suffix, human_prefixes, llm_prefixes, vote_prefixes, args.tie_break_prefix)
        for suffix in suffixes
    ]
    agreement_stats = [analyze_human_agreement(df, suffix, human_prefixes) for suffix in suffixes]

    print(f"Input: {args.input_csv}")
    print(f"Rows: {df.height}")
    print(f"Audited suffixes: {', '.join(suffixes)}")
    print(f"Human prefixes: {', '.join(human_prefixes)}")
    print(f"LLM prefixes: {', '.join(llm_prefixes)}")
    print(f"Vote prefixes for tie reconstruction: {', '.join(vote_prefixes)}")
    print(f"Tie-break prefix: {args.tie_break_prefix}")
    print("\nNote: provenance is support-based. A final majority can be supported by humans and LLMs simultaneously.")

    for stats in provenance_stats:
        print_provenance(stats)
    if len(provenance_stats) > 1:
        print_provenance(sum_provenance_stats(provenance_stats))

    for stats in agreement_stats:
        print_agreement(stats)

    if args.json_output:
        report = {
            "input_csv": str(args.input_csv),
            "rows": df.height,
            "suffixes": suffixes,
            "human_prefixes": human_prefixes,
            "llm_prefixes": llm_prefixes,
            "vote_prefixes": vote_prefixes,
            "tie_break_prefix": args.tie_break_prefix,
            "provenance": [asdict(stats) for stats in provenance_stats],
            "provenance_total": asdict(sum_provenance_stats(provenance_stats)),
            "human_agreement": [asdict(stats) for stats in agreement_stats],
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {args.json_output}")


if __name__ == "__main__":
    main()
