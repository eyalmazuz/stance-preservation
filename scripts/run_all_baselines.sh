#!/usr/bin/env bash

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

mkdir -p results

ALL_MODELS=(
  "bleu"
  "rouge1"
  "rouge2"
  "rougeL"
  "tf-idf"
  "emb"
  "llm"
  "nli"
)
MODELS=("${ALL_MODELS[@]}")
AGGREGATE_LEVELS=("sentence" "article")
LANGUAGES=("en" "he")
LABEL_PREFIX="majority"
CUSTOM_DATA=""
RESULTS_DIR="results"

declare -A INPUT_FILES=(
  ["en"]="data/datasets/english_normalized_majority.csv"
  ["he"]="data/datasets/hebrew_normalized_majority.csv"
)

declare -A LANGUAGE_NAMES=(
  ["en"]="English"
  ["he"]="Hebrew"
)

usage() {
  cat <<'EOF'
Usage: scripts/run_all_baselines.sh [options]

Options:
  --aggregate LEVEL       Run only one aggregate level: sentence or article.
  --language LANG         Run only one language: en or he.
  --data PATH             Use a custom input CSV instead of the default for the selected language(s).
  --results-dir PATH      Directory for prediction CSVs and run reports. Default: results.
  --prefix PREFIX         Label prefix to use, e.g. majority, annotator_A, annotator_B, GPT, Gemini.
  --models MODELS         Comma-separated or space-separated model list.
                          Example: --models bleu,rouge1,llm
                          Example: --models bleu rouge1 llm --language en
  -h, --help              Show this help.

No options preserves the default behavior: all models, both aggregate levels,
both languages, majority labels, and the default per-language CSVs.
EOF
}

die() {
  echo "Error: $*" >&2
  echo >&2
  usage >&2
  exit 2
}

contains_value() {
  local needle="$1"
  shift
  local value
  for value in "$@"; do
    [[ "$value" == "$needle" ]] && return 0
  done
  return 1
}

parse_models() {
  MODELS=()
  while [[ "$#" -gt 0 ]]; do
    IFS="," read -ra parts <<<"$1"
    local part
    for part in "${parts[@]}"; do
      [[ -n "$part" ]] && MODELS+=("$part")
    done
    shift
  done
  [[ "${#MODELS[@]}" -gt 0 ]] || die "--models requires at least one model"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --aggregate)
      [[ "$#" -ge 2 ]] || die "--aggregate requires a value"
      [[ "$2" == "sentence" || "$2" == "article" ]] || die "--aggregate must be sentence or article"
      AGGREGATE_LEVELS=("$2")
      shift 2
      ;;
    --language)
      [[ "$#" -ge 2 ]] || die "--language requires a value"
      [[ "$2" == "en" || "$2" == "he" ]] || die "--language must be en or he"
      LANGUAGES=("$2")
      shift 2
      ;;
    --data)
      [[ "$#" -ge 2 ]] || die "--data requires a path"
      CUSTOM_DATA="$2"
      shift 2
      ;;
    --results-dir)
      [[ "$#" -ge 2 ]] || die "--results-dir requires a value"
      RESULTS_DIR="$2"
      shift 2
      ;;
    --prefix)
      [[ "$#" -ge 2 ]] || die "--prefix requires a value"
      LABEL_PREFIX="$2"
      shift 2
      ;;
    --models)
      shift
      [[ "$#" -gt 0 ]] || die "--models requires at least one model"
      model_args=()
      while [[ "$#" -gt 0 && "$1" != --* ]]; do
        model_args+=("$1")
        shift
      done
      parse_models "${model_args[@]}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

for model in "${MODELS[@]}"; do
  contains_value "$model" "${ALL_MODELS[@]}" || die "unsupported model: $model"
done

if [[ -n "$CUSTOM_DATA" ]]; then
  [[ -f "$CUSTOM_DATA" ]] || die "custom data file does not exist: $CUSTOM_DATA"
  for language in "${LANGUAGES[@]}"; do
    INPUT_FILES[$language]="$CUSTOM_DATA"
  done
fi

run_one() {
  local aggregate_level="$1"
  local language="$2"
  local model="$3"
  local report_file="$4"
  local input_file="${INPUT_FILES[$language]}"
  local tmp_output
  local status

  tmp_output="$(mktemp)"

  local cmd=(
    "$PYTHON_BIN" main.py
    --input-file "$input_file"
    --language "$language"
    --aggregate-level "$aggregate_level"
    --model "$model"
    --label-prefix "$LABEL_PREFIX"
    --results-dir "$RESULTS_DIR"
  )

  if [[ "$model" == "llm" ]]; then
    cmd+=(--llm-concurrency 8)
  fi

  {
    echo
    echo "================================================================================"
    echo "Aggregate level: $aggregate_level"
    echo "Language: ${LANGUAGE_NAMES[$language]} ($language)"
    echo "Model: $model"
    echo "Label prefix: $LABEL_PREFIX"
    echo "Input: $input_file"
    echo "Command: ${cmd[*]}"
    echo "--------------------------------------------------------------------------------"
  } | tee -a "$report_file"

  "${cmd[@]}" >"$tmp_output" 2>&1
  status=$?

  cat "$tmp_output" | tee -a "$report_file"

  if [[ "$status" -eq 0 ]]; then
    echo "Status: OK" | tee -a "$report_file"
  else
    echo "Status: FAILED ($status)" | tee -a "$report_file"
  fi

  rm -f "$tmp_output"
  return 0
}

written_reports=()

for aggregate_level in "${AGGREGATE_LEVELS[@]}"; do
  mkdir -p "$RESULTS_DIR"
  report_file="$RESULTS_DIR/all_baselines_${aggregate_level}.txt"
  written_reports+=("$report_file")
  {
    echo "Stance preservation baseline results"
    echo "Aggregate level: $aggregate_level"
    echo "Generated at: $(date -Iseconds)"
    echo "Python: $PYTHON_BIN"
    echo "Models: ${MODELS[*]}"
    echo "Languages: ${LANGUAGES[*]}"
    echo "Label prefix: $LABEL_PREFIX"
    if [[ -n "$CUSTOM_DATA" ]]; then
      echo "Custom data: $CUSTOM_DATA"
    fi
  } >"$report_file"

  for language in "${LANGUAGES[@]}"; do
    for model in "${MODELS[@]}"; do
      run_one "$aggregate_level" "$language" "$model" "$report_file"
    done
  done
done

echo
echo "Wrote reports:"
for report_file in "${written_reports[@]}"; do
  echo "  $report_file"
done
echo
for report_file in "${written_reports[@]}"; do
  echo "$(basename "$report_file" .txt | sed 's/all_baselines_//') report:"
  cat "$report_file"
  echo
done
