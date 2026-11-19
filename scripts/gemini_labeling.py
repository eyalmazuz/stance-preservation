import argparse
import json
import os
import tempfile
import time

from typing import Any, Literal

import polars as pl

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("dotenv is not found falling back to environ.")


class LabelingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence: str
    sentence_topic: str
    sentence_stance: Literal["Favor", "Against", "Neutral"]


SYSTEM_INSTRUCTION = "You are an advanced NLP model specializing in stance detection."
TERMINAL_BATCH_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True, help="Path to the csv file we use for labeling")
    parser.add_argument("--save-path", type=str, required=True, help="Path to where to save the new CSV")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--batch-id",
        type=str,
        help="Existing Gemini Batch API job name or ID. Skips request creation and waits for this batch result.",
    )
    source_group.add_argument(
        "--jsonl-path",
        type=str,
        help="Path to a local Gemini Batch API output JSONL file. Skips request creation and batch polling.",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default="./data/prompts/en_labeling_prompt.txt",
        help="Path to the prompt used to label the data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Which Gemini model to use.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=30,
        help="How often to poll batch status when waiting for a Gemini batch.",
    )

    args = parser.parse_args()

    return args


def create_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def request_key(side: Literal["summary", "article"], row_index: int) -> str:
    return f"{side}-{row_index}"


def create_generate_content_request(sentence: str, prompt: str) -> dict[str, Any]:
    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=LabelingResponse.model_json_schema(),
    ).model_dump(exclude_none=True, by_alias=True)

    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt.format(sentence=sentence)}],
            }
        ],
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "generationConfig": generation_config,
    }


def create_batch_file(client: genai.Client, df: pl.DataFrame, prompt: str):
    requests: list[dict] = []
    for index, row in enumerate(df.iter_rows(named=True)):
        requests.append(
            {
                "key": request_key("summary", index),
                "request": create_generate_content_request(row["sentence_in_summary"], prompt),
            }
        )
        requests.append(
            {
                "key": request_key("article", index),
                "request": create_generate_content_request(row["best_match_sentences_from_article"], prompt),
            }
        )

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", encoding="utf-8", delete=False) as tmp_file:
        for record in requests:
            tmp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path = tmp_file.name

    try:
        return client.files.upload(
            file=temp_path,
            config=types.UploadFileConfig(display_name="gemini-labeling-batch-input", mime_type="jsonl"),
        )
    finally:
        os.unlink(temp_path)


def wait_for_batch(client: genai.Client, batch_id: str, poll_interval_seconds: int = 30):
    while True:
        batch_info = client.batches.get(name=batch_id)
        state = batch_info.state.name if batch_info.state else "JOB_STATE_UNSPECIFIED"
        print(f"Batch {batch_id} status: {state}", flush=True)

        if state in TERMINAL_BATCH_STATES:
            if state != "JOB_STATE_SUCCEEDED":
                raise RuntimeError(f"Batch {batch_id} ended with status: {state}. Error: {batch_info.error}")
            return batch_info

        time.sleep(poll_interval_seconds)


def iter_jsonl_lines(output_file_response):
    if isinstance(output_file_response, bytes):
        lines = output_file_response.decode("utf-8").splitlines()
    elif isinstance(output_file_response, str):
        lines = output_file_response.splitlines()
    elif hasattr(output_file_response, "iter_lines"):
        lines = output_file_response.iter_lines()
    else:
        lines = output_file_response

    for line in lines:
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        yield line


def existing_column_or_empty(df: pl.DataFrame, column: str, fallback_column: str) -> list[str]:
    if column in df.columns:
        return df[column].to_list()
    if fallback_column in df.columns:
        return df[fallback_column].to_list()
    return [""] * df.height


def get_result_key(result: dict[str, Any], line_index: int) -> str:
    key = result.get("key")
    if key is None:
        metadata = result.get("metadata") or {}
        key = metadata.get("key")
    if key is None:
        response_metadata = (result.get("response") or {}).get("metadata") or {}
        key = response_metadata.get("key")
    if key is None:
        side = "summary" if line_index % 2 == 0 else "article"
        key = request_key(side, line_index // 2)
        print(f"Result line {line_index + 1} did not include a key; falling back to input order as {key}.")
    return key


def extract_response_text(result: dict[str, Any]) -> str:
    response = result.get("response")
    if not response:
        raise ValueError(f"Batch result does not include a response: {result}")

    for candidate in response.get("candidates", []):
        content = candidate.get("content") or {}
        text_parts = [part["text"] for part in content.get("parts", []) if part.get("text")]
        if text_parts:
            return "".join(text_parts)

    raise ValueError(f"Batch result response does not include text output: {result}")


def add_batch_results_to_dataframe(output_file_response, df: pl.DataFrame) -> pl.DataFrame:
    row_count = df.height
    summary_topics = existing_column_or_empty(df, "Gemini_summary_topic", "summary_topic")
    summary_stances = existing_column_or_empty(df, "Gemini_summary_stance", "summary_stance")
    article_topics = existing_column_or_empty(df, "Gemini_article_topic", "article_topic")
    article_stances = existing_column_or_empty(df, "Gemini_article_stance", "article_stance")

    for line_index, line in enumerate(iter_jsonl_lines(output_file_response)):
        result = json.loads(line)
        unique_id = get_result_key(result, line_index)
        try:
            side, row_index_text = unique_id.split("-", 1)
            row_index = int(row_index_text)
        except ValueError:
            print(f"Skipping result with invalid key: {unique_id}")
            continue

        if side not in {"summary", "article"}:
            print(f"Skipping result with unknown side in key: {unique_id}")
            continue
        if row_index < 0 or row_index >= row_count:
            print(f"Skipping result with out-of-range row index: {unique_id}")
            continue
        if error := result.get("error"):
            print(f"Skipping failed batch request {unique_id}: {error}")
            continue

        content = extract_response_text(result)
        labeling = LabelingResponse.model_validate_json(content)

        if side == "summary":
            summary_topics[row_index] = labeling.sentence_topic
            summary_stances[row_index] = labeling.sentence_stance
        else:
            article_topics[row_index] = labeling.sentence_topic
            article_stances[row_index] = labeling.sentence_stance

    return df.with_columns(
        [
            pl.Series("Gemini_summary_topic", summary_topics),
            pl.Series("Gemini_summary_stance", summary_stances),
            pl.Series("Gemini_article_topic", article_topics),
            pl.Series("Gemini_article_stance", article_stances),
        ]
    )


def main() -> None:
    args = parse_args()
    df = pl.read_csv(args.csv_path)

    if args.jsonl_path:
        with open(args.jsonl_path, encoding="utf-8") as output_file:
            df = add_batch_results_to_dataframe(output_file, df)
    else:
        client = create_client()
        batch_id: str | None = args.batch_id
        if batch_id is None:
            with open(args.prompt_path) as fd:
                prompt = fd.read()

            batch_input_file = create_batch_file(client, df, prompt)
            batch_job = client.batches.create(
                model=args.model,
                src=batch_input_file.name,
                config={"display_name": "gemini-labeling-batch"},
            )
            if batch_job.name is None:
                raise RuntimeError("Gemini Batch API response did not include a batch job name.")
            batch_id = batch_job.name
            print(f"Created batch {batch_id}", flush=True)

        batch_info = wait_for_batch(client, batch_id, args.poll_interval_seconds)
        if batch_info.dest is None or batch_info.dest.file_name is None:
            raise RuntimeError(f"Batch {batch_id} completed without an output file.")

        output_file_response = client.files.download(file=batch_info.dest.file_name)
        df = add_batch_results_to_dataframe(output_file_response, df)

    df.write_csv(args.save_path)
    print(f"Saved merged CSV to {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
