import argparse
import json
import os
import tempfile
import time

from typing import Literal

import polars as pl

from openai import OpenAI
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True, help="Path to the csv file we use for labeling")
    parser.add_argument("--save-path", type=str, required=True, help="Path to where to save the new CSV")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--batch-id",
        type=str,
        help="Existing OpenAI Batch API job ID. Skips request creation and waits for this batch result.",
    )
    source_group.add_argument(
        "--jsonl-path",
        type=str,
        help="Path to a local Batch API output JSONL file. Skips request creation and batch polling.",
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
        default="gpt-5-mini-2025-08-07",
        choices=["gpt-5-nano-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-5-2025-08-07"],
        help="Which model to use.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=30,
        help="How often to poll batch status when waiting for an OpenAI batch.",
    )

    args = parser.parse_args()

    return args


def create_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORG_ID"),
        project=os.getenv("OPENAI_PROJECT_ID"),
    )


def create_batch_file(client: OpenAI, df: pl.DataFrame, prompt: str, model: str):
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "labeling_response",
            "strict": True,
            "schema": LabelingResponse.model_json_schema(),
        },
    }
    requests: list[dict] = []
    for index, row in enumerate(df.iter_rows(named=True)):
        summary_request = {
            "custom_id": f"summary-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an advanced NLP model specializing in stance detection."},
                    {
                        "role": "user",
                        "content": prompt.format(sentence=row["sentence_in_summary"]),
                    },
                ],
                "response_format": response_format,
            },
        }

        article_request = {
            "custom_id": f"article-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an advanced NLP model specializing in stance detection."},
                    {
                        "role": "user",
                        "content": prompt.format(sentence=row["best_match_sentences_from_article"]),
                    },
                ],
                "response_format": response_format,
            },
        }
        requests.append(summary_request)
        requests.append(article_request)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as tmp_file:
        for record in requests:
            tmp_file.write(json.dumps(record, ensure_ascii=False) + "\n")  # now dumps once
        temp_path = tmp_file.name

    with open(temp_path, "rb") as tmp_file:
        batch_input_file = client.files.create(file=tmp_file, purpose="batch")
    return batch_input_file


def wait_for_batch(client: OpenAI, batch_id: str, poll_interval_seconds: int = 30):
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}

    while True:
        batch_info = client.batches.retrieve(batch_id)
        print(f"Batch {batch_id} status: {batch_info.status}", flush=True)

        if batch_info.status in terminal_statuses:
            if batch_info.status != "completed":
                raise RuntimeError(f"Batch {batch_id} ended with status: {batch_info.status}")
            return batch_info

        time.sleep(poll_interval_seconds)


def iter_jsonl_lines(output_file_response):
    lines = output_file_response.iter_lines() if hasattr(output_file_response, "iter_lines") else output_file_response
    for line in lines:
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        yield line


def add_batch_results_to_dataframe(output_file_response, df: pl.DataFrame) -> pl.DataFrame:
    row_count = df.height
    summary_topics = df["summary_topic"].to_list() if "summary_topic" in df.columns else [""] * row_count
    summary_stances = df["summary_stance"].to_list() if "summary_stance" in df.columns else [""] * row_count
    article_topics = df["article_topic"].to_list() if "article_topic" in df.columns else [""] * row_count
    article_stances = df["article_stance"].to_list() if "article_stance" in df.columns else [""] * row_count

    for line in iter_jsonl_lines(output_file_response):
        result = json.loads(line)
        custom_id = result.get("custom_id", "")
        try:
            side, row_index_text = custom_id.split("-", 1)
            row_index = int(row_index_text)
        except ValueError:
            print(f"Skipping result with invalid custom_id: {custom_id}")
            continue

        if side not in {"summary", "article"}:
            print(f"Skipping result with unknown side in custom_id: {custom_id}")
            continue
        if row_index < 0 or row_index >= row_count:
            print(f"Skipping result with out-of-range row index: {custom_id}")
            continue
        if result.get("error"):
            print(f"Skipping failed batch request {custom_id}: {result['error']}")
            continue

        content = result["response"]["body"]["choices"][0]["message"]["content"]
        labeling = LabelingResponse.model_validate_json(content)

        if side == "summary":
            summary_topics[row_index] = labeling.sentence_topic
            summary_stances[row_index] = labeling.sentence_stance
        else:
            article_topics[row_index] = labeling.sentence_topic
            article_stances[row_index] = labeling.sentence_stance

    return df.with_columns(
        [
            pl.Series("GPT_summary_topic", summary_topics),
            pl.Series("GPT_summary_stance", summary_stances),
            pl.Series("GPT_article_topic", article_topics),
            pl.Series("GPT_article_stance", article_stances),
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
        batch_id = args.batch_id
        if batch_id is None:
            with open(args.prompt_path) as fd:
                prompt = fd.read()

            batch_input_file = create_batch_file(client, df, prompt, args.model)
            batch_job = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": "nightly eval job"},
            )
            batch_id = batch_job.id
            print(f"Created batch {batch_id}", flush=True)

        batch_info = wait_for_batch(client, batch_id, args.poll_interval_seconds)
        if batch_info.output_file_id is None:
            raise RuntimeError(f"Batch {batch_id} completed without an output file.")

        output_file_response = client.files.content(batch_info.output_file_id)
        df = add_batch_results_to_dataframe(output_file_response, df)

    df.write_csv(args.save_path)
    print(f"Saved merged CSV to {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
