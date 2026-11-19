#!/usr/bin/env python3
"""Standalone structured Gemini baseline mirroring scripts/structured_llm_baseline.py."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
import ot
import polars as pl
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from scipy.stats import kendalltau, pearsonr, spearmanr
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.data_utils import TextPair, process_data  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env", override=True)
except ImportError:
    pass


DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_PROMPT = ROOT_DIR / "data/prompts/structured_llm_emd_prompt.txt"
SYSTEM_INSTRUCTION = (
    "You are an expert stance annotator. Follow the supplied multi-step protocol exactly. "
    "Analyze source and summary independently and return only the requested structured output."
)
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
    parser.add_argument("--list-models", action="store_true", help="List available Gemini generateContent models and exit.")
    parser.add_argument("--run-suite", action="store_true", help="Run all 4 splits (full HE/EN + human-supported HE/EN).")
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument("--robustness-data-dir", type=Path, default=ROOT_DIR / "data/reference_robustness")
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--language", choices=["he", "en"])
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
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high", "none"], default="low")
    parser.add_argument("--max-output-tokens", type=int, default=16_000)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Process only the first N document-summary pairs.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore an existing audit file and start over.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first rendered prompt without API calls.")
    args = parser.parse_args()
    if not args.list_models and not args.run_suite:
        if args.input_file is None or args.language is None:
            parser.error("--input-file and --language are required unless --list-models or --run-suite is set")
        if args.output_csv is None:
            safe_model = args.model.replace("/", "_").replace(":", "_").replace("-", "_")
            args.output_csv = Path(f"results/{args.language}_structured_{safe_model}_baseline.csv")
        if args.audit_jsonl is None:
            safe_model = args.model.replace("/", "_").replace(":", "_").replace("-", "_")
            args.audit_jsonl = Path(f"results/{args.language}_structured_{safe_model}_baseline_audit.jsonl")
    return args


def create_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment or .env")
    return genai.Client(api_key=api_key)


def list_gemini_models() -> None:
    client = create_client()
    names = []
    for m in client.models.list():
        methods = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
        name = getattr(m, "name", "")
        if "gemini" in name.lower():
            names.append((name, getattr(m, "display_name", ""), list(methods)))
    for name, disp, methods in sorted(names):
        if not methods or any("generateContent" in str(a) for a in methods):
            print(f"{name}\t({disp})")


def document_id(pair: TextPair) -> str:
    payload = f"{pair.article}\0{pair.summary}".encode()
    return hashlib.sha256(payload).hexdigest()


def run_signature(model: str, prompt: str, language: str, thinking_level: str) -> str:
    payload = f"{model}\0{thinking_level}\0{language}\0{prompt}".encode()
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


def make_thinking_config(model: str, thinking_level: str) -> types.ThinkingConfig | None:
    if thinking_level == "none":
        return None
    # Gemini 3+ models use thinking_level; Gemini 2.5 models use thinking_budget
    if "gemini-2.5" in model.lower():
        budget_map = {"minimal": 0, "low": 1024, "medium": 4096, "high": 8192}
        return types.ThinkingConfig(thinking_budget=budget_map[thinking_level])
    level_map = {
        "minimal": types.ThinkingLevel.MINIMAL,
        "low": types.ThinkingLevel.LOW,
        "medium": types.ThinkingLevel.MEDIUM,
        "high": types.ThinkingLevel.HIGH,
    }
    return types.ThinkingConfig(thinking_level=level_map[thinking_level])


def call_model(
    client: genai.Client,
    model: str,
    prompt: str,
    thinking_level: str,
    max_output_tokens: int,
    expected_pairs: int,
    max_retries: int,
) -> tuple[DocumentJudgment | None, str, str | None]:
    thinking_cfg = make_thinking_config(model, thinking_level)
    safety_settings = [
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    ]
    config_kwargs = dict(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_json_schema=DocumentJudgment.model_json_schema(),
        safety_settings=safety_settings,
    )
    if thinking_cfg is not None:
        config_kwargs["thinking_config"] = thinking_cfg

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            fb = getattr(response, "prompt_feedback", None)
            block_reason = getattr(fb, "block_reason", None) if fb else None
            cands = getattr(response, "candidates", None) or []
            finish_reason = getattr(cands[0], "finish_reason", None) if cands else None
            raw_text = response.text
            if not raw_text:
                if block_reason is not None or (finish_reason is not None and "SAFETY" in str(finish_reason).upper() or "PROHIBITED" in str(finish_reason).upper()):
                    reason_str = str(block_reason or finish_reason)
                    resp_id = getattr(response, "response_id", None) or ""
                    return None, str(resp_id), reason_str
                raise RuntimeError(f"Empty response text from Gemini API (finish_reason={finish_reason}, prompt_feedback={fb}).")
            judgment = DocumentJudgment.model_validate_json(raw_text)
            validate_document_judgment(judgment, expected_pairs)
            resp_id = getattr(response, "response_id", None) or ""
            return judgment, str(resp_id), None
        except Exception as err:
            last_err = err
            if "thinking" in str(err).lower() and "thinking_config" in config_kwargs:
                config_kwargs.pop("thinking_config", None)
                continue
            if attempt < max_retries:
                time.sleep(min(2.0**attempt, 16.0))
    assert last_err is not None
    raise last_err


def evaluate_pair(
    client: genai.Client,
    pair: TextPair,
    language: str,
    model: str,
    template: str,
    signature: str,
    thinking_level: str,
    max_output_tokens: int,
    max_retries: int,
) -> dict[str, object]:
    prompt = build_prompt(template, pair, language)
    judgment, response_id, block_reason = call_model(
        client=client,
        model=model,
        prompt=prompt,
        thinking_level=thinking_level,
        max_output_tokens=max_output_tokens,
        expected_pairs=len(pair.article_data),
        max_retries=max_retries,
    )
    if judgment is None:
        scores = {
            "structured_llm_emd_preds": 0.0,
            "structured_llm_argmax_preds": 0.0,
            "structured_llm_topic_coverage": 0.0,
            "structured_llm_label_consistency": 0.0,
        }
        judgment_dump: dict[str, object] = {"pairs": []}
    else:
        scores = score_judgment(judgment)
        judgment_dump = judgment.model_dump(mode="json")
    return {
        "document_id": document_id(pair),
        "run_signature": signature,
        "response_id": response_id,
        "model": model,
        "thinking_level": thinking_level,
        "language": language,
        "refused": judgment is None,
        "block_reason": block_reason or "",
        "article": pair.article,
        "summary": pair.summary,
        "score": pair.score,
        **scores,
        "judgment": judgment_dump,
    }


def load_completed(path: Path, signature: str) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    completed = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
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
        "refused",
        "block_reason",
        "response_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            record = records[document_id(pair)]
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def print_correlations(pairs: list[TextPair], records: dict[str, dict[str, object]]) -> None:
    refused_ids = {document_id(p) for p in pairs if records[document_id(p)].get("refused")}
    non_refused = [p for p in pairs if document_id(p) not in refused_ids]
    cov_all = np.mean([float(records[document_id(pair)]["structured_llm_topic_coverage"]) for pair in pairs])
    cov_ok = np.mean([float(records[document_id(pair)]["structured_llm_topic_coverage"]) for pair in non_refused]) if non_refused else 0.0
    cons_ok = np.mean([float(records[document_id(pair)]["structured_llm_label_consistency"]) for pair in non_refused]) if non_refused else 0.0
    print(f"\nRefused by API: {len(refused_ids)}/{len(pairs)} | Mean topic coverage: {100*cov_all:.1f}% (all) / {100*cov_ok:.1f}% (non-refused) | Label consistency (non-refused): {100*cons_ok:.1f}%")
    if len(pairs) < 2:
        print("Skipping correlations (fewer than 2 pairs).")
        return
    for subset_name, subset in [("ALL (refusal=0.0 sentinel)", pairs), ("NON-REFUSED ONLY", non_refused)]:
        if subset_name == "NON-REFUSED ONLY" and not refused_ids:
            continue
        print(f"--- {subset_name} (N={len(subset)}) ---")
        gold = np.asarray([pair.score for pair in subset], dtype=float)
        for prediction_column in ("structured_llm_emd_preds", "structured_llm_argmax_preds"):
            predictions = np.asarray([records[document_id(pair)][prediction_column] for pair in subset], dtype=float)
            print(f"{prediction_column}: ", end="")
            parts = []
            for name, function in (("r", pearsonr), ("rho", spearmanr), ("tau", kendalltau)):
                statistic, pvalue = function(predictions, gold)
                parts.append(f"{name}={statistic:.3f} (p={pvalue:.3f})")
            print(", ".join(parts))


def run_single(
    client: genai.Client,
    template: str,
    input_file: Path,
    language: str,
    model: str,
    thinking_level: str,
    output_csv: Path,
    audit_jsonl: Path,
    label_prefix: str,
    reference_score_method: str,
    concurrency: int,
    max_output_tokens: int,
    max_retries: int,
    limit: int | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    pairs = process_data(
        pl.read_csv(input_file),
        label_prefix,
        reference_score_method,
    )
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError(f"Input {input_file} produced no document-summary pairs.")

    if dry_run:
        print(build_prompt(template, pairs[0], language))
        return

    signature = run_signature(model, template, language, thinking_level)
    if overwrite and audit_jsonl.exists():
        audit_jsonl.unlink()
    completed = load_completed(audit_jsonl, signature)
    pending = [pair for pair in pairs if document_id(pair) not in completed]
    print(
        f"\n=== [{language.upper()}] {input_file.name} | Model: {model} (thinking={thinking_level}) | "
        f"Total: {len(pairs)} | Pending: {len(pending)} ===",
        flush=True,
    )

    if pending:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(pending))) as executor:
            futures = {
                executor.submit(
                    evaluate_pair,
                    client,
                    pair,
                    language,
                    model,
                    template,
                    signature,
                    thinking_level,
                    max_output_tokens,
                    max_retries,
                ): pair
                for pair in pending
            }
            failures = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{model} ({language})"):
                pair = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    failures.append((document_id(pair), error))
                    print(f"\nFailed document {document_id(pair)[:12]}: {type(error).__name__}: {error}", flush=True)
                    continue
                completed[str(record["document_id"])] = record
                append_audit_record(audit_jsonl, record)

        if failures:
            raise RuntimeError(
                f"{len(failures)} document(s) failed in {input_file.name}; successful documents were checkpointed. Rerun to resume."
            )

    missing = [document_id(pair) for pair in pairs if document_id(pair) not in completed]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} document judgments after evaluation for {input_file.name}.")
    write_predictions(output_csv, pairs, completed)
    print_correlations(pairs, completed)
    print(f"Wrote {output_csv}", flush=True)
    print(f"Wrote {audit_jsonl}", flush=True)


def main() -> None:
    args = parse_args()
    if args.list_models:
        list_gemini_models()
        return

    template = args.prompt_file.read_text(encoding="utf-8")
    client = create_client()

    if args.run_suite:
        safe = args.model.replace("/", "_").replace(":", "_").replace("-", "_")
        res_base = args.results_dir
        rob_data = args.robustness_data_dir
        jobs = [
            (
                ROOT_DIR / "data/datasets/hebrew_normalized_majority.csv",
                "he",
                res_base / f"he_structured_{safe}_baseline.csv",
                res_base / f"he_structured_{safe}_baseline_audit.jsonl",
            ),
            (
                ROOT_DIR / "data/datasets/english_normalized_majority.csv",
                "en",
                res_base / f"en_structured_{safe}_baseline.csv",
                res_base / f"en_structured_{safe}_baseline_audit.jsonl",
            ),
            (
                rob_data / "hebrew_human_supported_robustness.csv",
                "he",
                res_base / "reference_robustness" / f"he_structured_{safe}_baseline__hebrew_human_supported_robustness.csv",
                res_base / "reference_robustness" / f"he_structured_{safe}_baseline__hebrew_human_supported_robustness_audit.jsonl",
            ),
            (
                rob_data / "english_human_supported_robustness.csv",
                "en",
                res_base / "reference_robustness" / f"en_structured_{safe}_baseline__english_human_supported_robustness.csv",
                res_base / "reference_robustness" / f"en_structured_{safe}_baseline__english_human_supported_robustness_audit.jsonl",
            ),
        ]
        for input_file, lang, out_csv, audit_jsonl in jobs:
            run_single(
                client=client,
                template=template,
                input_file=input_file,
                language=lang,
                model=args.model,
                thinking_level=args.thinking_level,
                output_csv=out_csv,
                audit_jsonl=audit_jsonl,
                label_prefix=args.label_prefix,
                reference_score_method=args.reference_score_method,
                concurrency=args.concurrency,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                limit=args.limit,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        print("\nALL 4 SUITE RUNS COMPLETE!", flush=True)
        return

    run_single(
        client=client,
        template=template,
        input_file=args.input_file,
        language=args.language,
        model=args.model,
        thinking_level=args.thinking_level,
        output_csv=args.output_csv,
        audit_jsonl=args.audit_jsonl,
        label_prefix=args.label_prefix,
        reference_score_method=args.reference_score_method,
        concurrency=args.concurrency,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        limit=args.limit,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
