# Structured GPT-5-mini stance baseline

This standalone baseline addresses the reviewer request for an LLM comparison that mirrors the stance-aware
pipeline instead of directly assigning a scalar quality score.

For every document-summary pair, it gives GPT-5-mini the full source, full summary, and existing aligned sentence
pairs. The model returns structured topic and stance judgments for each pair. It does **not** return a preservation
score. The script then:

1. filters pairs that the LLM marks as having different semantic stance targets;
2. normalizes the independently predicted Against/Neutral/Favor distributions;
3. computes ordinal Earth Mover's Distance using the same stance cost matrix as the proposed method;
4. averages distances across comparable pairs and converts them to a preservation score (`2 - mean EMD`);
5. also reports hard argmax preservation, topic coverage, and label/distribution consistency.

The prompt never includes gold topic or stance labels. Gold annotations are used only after prediction to construct
the reference preservation score for correlation analysis.

## Dry run

Inspect the first rendered English prompt without making an API call:

```bash
.venv/bin/python scripts/structured_llm_baseline.py \
  --input-file data/datasets/english_normalized_majority.csv \
  --language en \
  --limit 1 \
  --dry-run
```

## Full evaluation

```bash
.venv/bin/python scripts/structured_llm_baseline.py \
  --input-file data/datasets/english_normalized_majority.csv \
  --language en \
  --concurrency 4

.venv/bin/python scripts/structured_llm_baseline.py \
  --input-file data/datasets/hebrew_normalized_majority.csv \
  --language he \
  --concurrency 4
```

The default model is the pinned `gpt-5-mini-2025-08-07` snapshot. Authentication uses `OPENAI_API_KEY`, with
optional `OPENAI_ORG` and `OPENAI_PROJECT` environment variables.

Default outputs are:

- `results/{language}_structured_llm_baseline.csv`: one row per document-summary pair with scores;
- `results/{language}_structured_llm_baseline_audit.jsonl`: raw structured judgments, evidence, API response IDs,
  and checkpoint metadata.

The JSONL file is a resumable checkpoint. Re-running an interrupted command skips completed documents when the
model, language, and prompt signature match. Use `--overwrite` to start a fresh run.

## Primary reported result

Use `structured_llm_emd_preds` as the main structured LLM baseline. The script prints Pearson, Spearman, and Kendall
correlations and p-values against the selected reference score. `structured_llm_argmax_preds` is a useful secondary
ablation showing the effect of retaining versus discarding probability information.
