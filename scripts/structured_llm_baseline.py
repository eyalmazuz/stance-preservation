#!/usr/bin/env python3
"""Standalone structured GPT baseline that mirrors the stance-aware EMD pipeline."""

import argparse
import csv
import hashlib
import json
import os
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
import ot
import polars as pl

from openai import OpenAI
from pydantic import BaseModel, Field
from scipy.stats import kendalltau, pearsonr, spearmanr
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.data_utils import TextPair, process_data  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_PROMPT = Path("data/prompts/structured_llm_emd_prompt.txt")
STANCE_LABELS = ("Against", "Neutral", "Favor")
STANCE_COST = np.asarray(
    [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


class StanceDistribution(BaseModel):
    against: float = Field(ge=0.0, le=1.0, description="Probability of Against stance.")
    neutral: float = Field(ge=0.0, le=1.0, description="Probability of Neutral stance.")
    favor: float = Field(ge=0.0, le=1.0, description="Probability of Favor stance.")


class PairJudgment(BaseModel):
    pair_id: int = Field(ge=0, description="The exact pair ID supplied in the prompt.")
    source_topic: str = Field(max_length=120, description="Concise stance target extracted from the source sentence.")
    summary_topic: str = Field(
        max_length=120,
        description="Concise stance target extracted from the summary sentence.",
    )
    topics_match: bool = Field(description="Whether both topics are semantically the same stance target.")
    comparison_topic: str = Field(
        max_length=120,
        description='Canonical shared topic, or "NONE" if topics do not match.',
    )
    source_stance: Literal["Against", "Neutral", "Favor"]
    summary_stance: Literal["Against", "Neutral", "Favor"]
    source_distribution: StanceDistribution
    summary_distribution: StanceDistribution
    source_evidence: str = Field(
        max_length=240,
        description="Short evidence span or concise justification from the source.",
    )
    summary_evidence: str = Field(
        max_length=240,
        description="Short evidence span or concise justification from the summary.",
    )


class DocumentJudgment(BaseModel):
    pairs: list[PairJudgment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--language", choices=["he", "en"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--audit-jsonl", type=Path)
    parser.add_argument("--label-prefix", default="majority")
    parser.add_argument(
        "--reference-score-method",
        choices=["exact", "conditional-soft", "coverage-soft"],
        default="exact",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default="low")
    parser.add_argument("--max-output-tokens", type=int, default=16_000)
    parser.add_argument("--limit", type=int, help="Process only the first N document-summary pairs.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore an existing audit file and start over.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first rendered prompt without API calls.")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    if args.output_csv is None:
        args.output_csv = Path(f"results/{args.language}_structured_llm_baseline.csv")
    if args.audit_jsonl is None:
        args.audit_jsonl = Path(f"results/{args.language}_structured_llm_baseline_audit.jsonl")
    return args


def document_id(pair: TextPair) -> str:
    payload = f"{pair.article}\0{pair.summary}".encode()
    return hashlib.sha256(payload).hexdigest()


def run_signature(model: str, prompt: str, language: str) -> str:
    payload = f"{model}\0{language}\0{prompt}".encode()
    return hashlib.sha256(payload).hexdigest()


def render_aligned_pairs(pair: TextPair) -> str:
    if len(pair.article_data) != len(pair.summary_data):
        raise ValueError("Aligned source and summary sentence lists must have equal length.")
    blocks = []
    for pair_id, (source, summary) in enumerate(zip(pair.article_data, pair.summary_data, strict=True)):
        blocks.append(f"[PAIR {pair_id}]\nSOURCE SENTENCE: {source['text']}\nSUMMARY SENTENCE: {summary['text']}")
    return "\n\n".join(blocks)


def build_prompt(template: str, pair: TextPair, language: str) -> str:
    language_name = "Hebrew" if language == "he" else "English"
    return template.format(
        language=language_name,
        article=pair.article,
        summary=pair.summary,
        aligned_pairs=render_aligned_pairs(pair),
    )


def distribution_vector(distribution: StanceDistribution) -> np.ndarray:
    values = np.asarray(
        [distribution.against, distribution.neutral, distribution.favor],
        dtype=np.float64,
    )
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("A stance distribution must have positive finite mass.")
    return values / total


def argmax_label(distribution: StanceDistribution) -> str:
    return STANCE_LABELS[int(np.argmax(distribution_vector(distribution)))]


def validate_document_judgment(judgment: DocumentJudgment, expected_pairs: int) -> None:
    pair_ids = [item.pair_id for item in judgment.pairs]
    if sorted(pair_ids) != list(range(expected_pairs)):
        raise ValueError(f"Expected pair IDs 0..{expected_pairs - 1}, received {sorted(pair_ids)}")
    for item in judgment.pairs:
        distribution_vector(item.source_distribution)
        distribution_vector(item.summary_distribution)
        if item.topics_match and item.comparison_topic.strip().upper() == "NONE":
            raise ValueError(f"Pair {item.pair_id} marks matching topics but has no comparison topic.")
        if not item.topics_match and item.comparison_topic.strip().upper() != "NONE":
            raise ValueError(f"Pair {item.pair_id} marks mismatched topics but supplies a comparison topic.")


def score_judgment(judgment: DocumentJudgment) -> dict[str, float]:
    comparable = [item for item in judgment.pairs if item.topics_match]
    if not judgment.pairs or not comparable:
        return {
            "structured_llm_emd_preds": 0.0,
            "structured_llm_argmax_preds": 0.0,
            "structured_llm_topic_coverage": 0.0,
            "structured_llm_label_consistency": 0.0,
        }

    emd_distances = []
    exact_matches = []
    label_consistency = []
    for item in comparable:
        source = distribution_vector(item.source_distribution)
        summary = distribution_vector(item.summary_distribution)
        emd_distances.append(float(ot.emd2(source, summary, STANCE_COST)))
        source_argmax = argmax_label(item.source_distribution)
        summary_argmax = argmax_label(item.summary_distribution)
        exact_matches.append(float(source_argmax == summary_argmax))
        label_consistency.extend(
            [
                float(source_argmax == item.source_stance),
                float(summary_argmax == item.summary_stance),
            ]
        )

    return {
        "structured_llm_emd_preds": 2.0 - float(np.mean(emd_distances)),
        "structured_llm_argmax_preds": float(np.mean(exact_matches)),
        "structured_llm_topic_coverage": len(comparable) / len(judgment.pairs),
        "structured_llm_label_consistency": float(np.mean(label_consistency)),
    }


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    reasoning_effort: str,
    max_output_tokens: int,
    expected_pairs: int,
) -> tuple[DocumentJudgment, str]:
    response = client.responses.parse(
        model=model,
        instructions=(
            "You are an expert stance annotator. Follow the supplied multi-step protocol exactly. "
            "Analyze source and summary independently and return only the requested structured output."
        ),
        input=prompt,
        text_format=DocumentJudgment,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        store=False,
    )
    if response.output_parsed is None:
        raise RuntimeError("The model returned no parsed output, possibly because it refused the request.")
    validate_document_judgment(response.output_parsed, expected_pairs)
    return response.output_parsed, response.id


def evaluate_pair(
    client: OpenAI,
    pair: TextPair,
    language: str,
    model: str,
    template: str,
    signature: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> dict[str, object]:
    prompt = build_prompt(template, pair, language)
    judgment, response_id = call_model(
        client=client,
        model=model,
        prompt=prompt,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        expected_pairs=len(pair.article_data),
    )
    scores = score_judgment(judgment)
    return {
        "document_id": document_id(pair),
        "run_signature": signature,
        "response_id": response_id,
        "model": model,
        "language": language,
        "article": pair.article,
        "summary": pair.summary,
        "score": pair.score,
        **scores,
        "judgment": judgment.model_dump(mode="json"),
    }


def load_completed(path: Path, signature: str) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    completed = {}
    # Iterate physical JSONL records. str.splitlines() also splits Unicode line
    # separators that can legitimately occur inside a JSON string.
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from error
            if record.get("run_signature") == signature:
                completed[str(record["document_id"])] = record
    return completed


def append_audit_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_predictions(path: Path, pairs: list[TextPair], records: dict[str, dict[str, object]]) -> None:
    fieldnames = [
        "article",
        "summary",
        "score",
        "structured_llm_emd_preds",
        "structured_llm_argmax_preds",
        "structured_llm_topic_coverage",
        "structured_llm_label_consistency",
        "response_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            record = records[document_id(pair)]
            writer.writerow({field: record[field] for field in fieldnames})


def print_correlations(pairs: list[TextPair], records: dict[str, dict[str, object]]) -> None:
    gold = np.asarray([pair.score for pair in pairs], dtype=float)
    for prediction_column in ("structured_llm_emd_preds", "structured_llm_argmax_preds"):
        predictions = np.asarray([records[document_id(pair)][prediction_column] for pair in pairs], dtype=float)
        print(f"\n{prediction_column}")
        for name, function in (("Pearson", pearsonr), ("Spearman", spearmanr), ("Kendall", kendalltau)):
            statistic, pvalue = function(predictions, gold)
            print(f"{name}- corr: {statistic:.3f}, p-value {pvalue:.3f}")


def main() -> None:
    args = parse_args()
    template = args.prompt_file.read_text(encoding="utf-8")
    pairs = process_data(
        pl.read_csv(args.input_file),
        args.label_prefix,
        args.reference_score_method,
    )
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("The input produced no document-summary pairs.")

    if args.dry_run:
        print(build_prompt(template, pairs[0], args.language))
        return

    signature = run_signature(args.model, template, args.language)
    if args.overwrite and args.audit_jsonl.exists():
        args.audit_jsonl.unlink()
    completed = load_completed(args.audit_jsonl, signature)
    pending = [pair for pair in pairs if document_id(pair) not in completed]

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization=os.environ.get("OPENAI_ORG"),
        project=os.environ.get("OPENAI_PROJECT"),
    )
    with ThreadPoolExecutor(max_workers=min(args.concurrency, max(1, len(pending)))) as executor:
        futures = {
            executor.submit(
                evaluate_pair,
                client,
                pair,
                args.language,
                args.model,
                template,
                signature,
                args.reasoning_effort,
                args.max_output_tokens,
            ): pair
            for pair in pending
        }
        failures = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Structured LLM documents"):
            pair = futures[future]
            try:
                record = future.result()
            except Exception as error:  # Continue checkpointing other concurrent successes.
                failures.append((document_id(pair), error))
                print(f"\nFailed document {document_id(pair)[:12]}: {type(error).__name__}: {error}")
                continue
            completed[str(record["document_id"])] = record
            append_audit_record(args.audit_jsonl, record)

    if failures:
        raise RuntimeError(
            f"{len(failures)} document(s) failed; successful documents were checkpointed. Rerun to resume."
        )

    missing = [document_id(pair) for pair in pairs if document_id(pair) not in completed]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} document judgments after evaluation.")
    write_predictions(args.output_csv, pairs, completed)
    print_correlations(pairs, completed)
    print(f"\nWrote {args.output_csv}")
    print(f"Wrote {args.audit_jsonl}")


if __name__ == "__main__":
    main()
