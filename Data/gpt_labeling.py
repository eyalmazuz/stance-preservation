import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import json
import time
import pandas as pd
import re
import io
import ast

def clean_article_sentence(cell):
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], tuple):
            return parsed[0][0]  # just return the sentence
    except:
        pass
    return cell

def quote_json_keys(block):
    block = re.sub(r'(?<=\n|{)\s*([a-zA-Z_][\w\s]*?)\s*:', r'"\1":', block)
    block = block.replace("\\'", "'")
    block = block.replace('\\"', '"')
    block = block.replace('\\\\', '\\')
    block = re.sub(r',\s*}', '}', block)
    return block


# def quote_json_keys(block):
#     return re.sub(r'(?<=\n|{)\s*([a-zA-Z_][\w\s]*?)\s*:', r'"\1":', block)

def quote_json_keys(block):
    # Ensure keys are quoted
    block = re.sub(r'(?<=\n|{)\s*([a-zA-Z_][\w\s]*?)\s*:', r'"\1":', block)

    # Remove bad backslash escapes like \' (keep \" because that's valid in JSON)
    block = block.replace("\\'", "'")

    return block

def parse_batch_output(file_obj):
    results = []
    for i, line in enumerate(file_obj.iter_lines()):
        if not line:
            continue
        try:
            response = json.loads(line)
            content = response['response']['body']['choices'][0]['message']['content']
            json_blocks = re.findall(r"```json\s*([\s\S]+?)\s*```", content)
            if not json_blocks:
                print(f"⚠️ No JSON block in object {i}, skipping.")
                continue
            for block in json_blocks:
                try:
                    safe_block = quote_json_keys(block)
                    data = json.loads(safe_block)
                    idx = data.get("index")
                    # topic = data.get("summary topic gpt")
                    # stance = data.get("summary stance gpt")
                    topic = data.get("article topic gpt")
                    stance = data.get("article stance gpt")
                    if idx is not None and topic and stance:
                        results.append({
                            "index": idx,
                            # "summary topic gpt": topic.strip(),
                            # "summary stance gpt": stance.strip()
                            "article topic gpt": topic.strip(),
                            "article stance gpt": stance.strip()
                        })
                    else:
                        print(f"✗ Incomplete data in object {i}: {data}")
                except json.JSONDecodeError as e:
                    print(f"✗ JSON decode error in object {i}: {e}")
                    print("Block content:", block[:300])
                    continue
        except Exception as e:
            print(f"✗ Unexpected error parsing object {i}: {e}")
            continue
    print(f"\n✅ Done. Parsed {len(results)} valid summary entries.")
    return results



def merge_results_into_csv(original_csv_path, parsed_results, output_csv_path):
    df = pd.read_csv(original_csv_path)
    for col in ['article topic gpt', 'article stance gpt']:
        if col not in df.columns:
            df[col] = ""

    for row in parsed_results:
        idx = row.get("index")
        if idx is not None and 0 <= idx < len(df):
            df.at[idx, 'article topic gpt'] = row.get("article topic gpt", "")
            df.at[idx, 'article stance gpt'] = row.get("article stance gpt", "")

    # # Make sure new columns exist
    # for col in ['summary topic gpt', 'summary stance gpt']:
    #     df[col] = ""

    # for row in parsed_results:
    #     idx = row.get("index")
    #     if idx is not None and 0 <= idx < len(df):
    #         df.at[idx, 'summary topic gpt'] = row.get("summary topic gpt", "")
    #         df.at[idx, 'summary stance gpt'] = row.get("summary stance gpt", "")

    df.to_csv(output_csv_path, index=False)


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID")
)

def get_completion(prompt, model="gpt-4o-mini"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message.content

def get_completion_from_messages(messages, model="gpt-4o-mini", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, 
    )
    return response.choices[0].message.content


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data


if __name__ == "__main__":
    # prompt = f"""
    #         "עבור המשפט הבא: "הנתונים מראים כי בקרב גברים אלה נצפתה ירידה של 52,
              
    #         ,קבע מה הנושא שלו (בקצרה, בין מילה לארבע מילים אם צריך)
    #         לאחר מכן בדוק מה העמדה של המשפט הזה כלפי הנושא הזה.
    #         האם הוא תומך בנושא, מתנגד לו או נייטרלי כלפיו?

    #         זכור כי *עמדה* היא *לא* סנטימנט.
    #         כלומר, אם הסנטימנט הוא שלילי זה לא אומר שהעמדה היא נגד. או אם הסנטימנט הוא חיובי זה לא אומר שהעמדה היא בעד.
    #         העמדות האפשריות: [בעד, נגד, נייטרלי]
            
    #         ### **Output Format:**
    #             ```json
    #             {{
    #                 Sentence in Summary: הנתונים מראים כי בקרב גברים אלה נצפתה ירידה של 52
    #                 summary topic gpt: "your topic here"
    #                 summary stance gpt: "your stance here"
    #             }}
    #             ```
    #         """
    # response = get_completion(prompt)
    # print(response)

    # ---------------------------------------------------------- JSONL creation ----------------------------------------------------------
    # data = pd.read_csv('./Data/datasets/all_data.csv')
    # data['Best Match Sentences From Article'] = data['Best Match Sentences From Article'].apply(clean_article_sentence)

    # requests = []
    # BATCH_SIZE = 5

    # for i in range(0, len(data), BATCH_SIZE):
    #     batch_df = data.iloc[i:i+BATCH_SIZE]
    #     combined_prompt = ""
    #     for j, row in batch_df.iterrows():
    #         sentence = row['Best Match Sentences From Article']
    #         if isinstance(sentence, str):
    #             sentence = sentence.replace('\\', '')
    #             sentence = sentence.replace('"', "'")
    #             sentence = sentence.replace('\n', ' ')
    #         combined_prompt += f"""
    #         עבור המשפט הבא: {sentence}

    #         ,קבע מה הנושא שלו (בקצרה, בין מילה לארבע מילים אם צריך)
    #         לאחר מכן בדוק מה העמדה של המשפט הזה כלפי הנושא הזה.
    #         האם הוא תומך בנושא, מתנגד לו או נייטרלי כלפיו?

    #         זכור כי *עמדה* היא *לא* סנטימנט.
    #         כלומר, אם הסנטימנט הוא שלילי זה לא אומר שהעמדה היא נגד. או אם הסנטימנט הוא חיובי זה לא אומר שהעמדה היא בעד.
    #         העמדות האפשריות: [בעד, נגד, נייטרלי]

    #         ### Output Format:
    #         ```json
    #         {{
    #             index: {j},
    #             sentence: "{sentence}",
    #             article topic gpt: "your topic here",
    #             article stance gpt: "your stance here"
    #         }}
    #         ```
    #         """

    #         # combined_prompt += f"""
    #         # עבור המשפט הבא: {row['Sentence in Summary']}

    #         # ,קבע מה הנושא שלו (בקצרה, בין מילה לארבע מילים אם צריך)
    #         # לאחר מכן בדוק מה העמדה של המשפט הזה כלפי הנושא הזה.
    #         # האם הוא תומך בנושא, מתנגד לו או נייטרלי כלפיו?

    #         # זכור כי *עמדה* היא *לא* סנטימנט.
    #         # כלומר, אם הסנטימנט הוא שלילי זה לא אומר שהעמדה היא נגד. או אם הסנטימנט הוא חיובי זה לא אומר שהעמדה היא בעד.
    #         # העמדות האפשריות: [בעד, נגד, נייטרלי]

    #         # ### Output Format:
    #         # ```json
    #         # {{
    #         #     index: {j},
    #         #     sentence: "{row['Sentence in Summary']}",
    #         #     summary topic gpt: "your topic here",
    #         #     summary stance gpt: "your stance here"
    #         # }}
    #         # ```
    #         # """

    #     request = {
    #         "custom_id": f"batch-{i}",
    #         "method": "POST",
    #         "url": "/v1/chat/completions",
    #         "body": {
    #             "model": "gpt-4o-mini",
    #             "messages": [
    #                 {"role": "system", "content": "You are an advanced NLP model specializing in stance detection."},
    #                 {"role": "user", "content": combined_prompt}
    #             ],
    #             "max_tokens": 2000
    #         }
    #     }
    #     requests.append(request)
        
        

    # # write to jsonl file
    # with open('./Data/requests/gpt_label2.jsonl', 'w', encoding='utf-8') as file:
    #     for request in requests:
    #         json_str = json.dumps(request, ensure_ascii=False)
    #         file.write(json_str + '\n')

    # # requests = read_jsonl('./Data/requests/gpt_label2.jsonl')
    # # print(requests)

    # ---------------------------------------------------------- Batch creation ----------------------------------------------------------
    # # Upload your batch input file
    # batch_input_file = client.files.create(
    #     file=open("./Data/requests/gpt_label2.jsonl", "rb"),
    #     purpose="batch"
    # )

    # print(f"Batch input file: {batch_input_file}")

    # # Create the batch
    # batch_input_file_id = batch_input_file.id
    # batch = client.batches.create(
    #     input_file_id=batch_input_file_id,
    #     endpoint="/v1/chat/completions",
    #     completion_window="24h",
    #     metadata={"description": "nightly eval job"}
    # )

    # batch_id = batch.id
    # print(f"Batch created with ID: {batch_id}")
    # ---------------------------------------------------------- Batch status check ----------------------------------------------------------

    # Wait for batch to complete
    batch_info = client.batches.retrieve('batch_6878a95d7170819084173e068c4bf2e9')
    print(batch_info)
    # batch_686d0ff44f3c8190a82c65df886cb890 - art
    # batch_686cfabb245c81909ef66597a94c5292 - sum
    # Check if batch completed successfully
    if batch_info.status == "completed":
        output_file_id = batch_info.output_file_id
        output_file_response = client.files.content(output_file_id)

        parsed_results = parse_batch_output(output_file_response)

        # After summary batch
        # merge_results_into_csv("./Data/datasets/all_data.csv", parsed_results, "temp_with_summary.csv")

        # After article batch
        merge_results_into_csv("temp_with_summary.csv", parsed_results, "./Data/datasets/all_data_with_gpt_labels.csv")


    # # Check for errors 
    # error_file_id = "file-LLsMjDt7j3aGSHuVpo2uEu"
    # error_response = client.files.content(error_file_id)
    # print(error_response.text)  # This will show why the requests failed


    # ---------------------------------------------------------- Batch cancel / list ----------------------------------------------------------

    # Cancel a batch
    # client.batches.cancel('batch_67ee395e02c0819091c5125d17dc47c1')

    # Get a list of all batches
    # client.batches.list(limit=10)