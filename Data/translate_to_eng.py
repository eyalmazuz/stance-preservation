import os
import json
import time
import sys
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID"),
)

# ---------------- Config ----------------
IN_PATH = "./Data/datasets/all_data.csv"
OUT_PATH = "./Data/datasets/translated_to_english.csv"
CLEAN_OUT_PATH = "./Data/datasets/translated_to_english_clean.csv"
CHECKPOINT_EVERY = 20
MODEL = "gpt-4o-mini"

FIELDS_OTHER = [
    "Sentence in Summary",
    "Best Match Sentences From Article",
    "summary topic",
    "summary stance",
    "article topic",
    "article stance",
]

PROMPT = """
Translate the following JSON values from Hebrew to English.
Return a JSON object with the exact same keys.
Do not include markdown, comments, or explanations.
If a value is already in English, return it unchanged.

JSON:
{json_payload}
"""

# -------------- Helpers -----------------
def clean_value(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    return s if s != "" else ""

CHUNK_SIZE = 1200  # safe for gpt-4o-mini input

def chunk_text(text, size=CHUNK_SIZE):
    return [text[i:i+size] for i in range(0, len(text), size)]

def safe_translate_field(key, value):
    if not value:
        return value

    # Small enough → normal path
    if len(value) <= CHUNK_SIZE:
        return translate_json({key: value})[key]

    print(f"🧩 Chunking {key} ({len(value)} chars)")

    chunks = chunk_text(value, CHUNK_SIZE)
    translated_chunks = []

    for i, chunk in enumerate(chunks):
        try:
            part = translate_json({key: chunk})[key]
            translated_chunks.append(part)
        except Exception as e:
            print(f"❌ Failed chunk {i+1}/{len(chunks)} for {key}: {e}")
            translated_chunks.append("")  # keep alignment

    return " ".join(translated_chunks)



import re

def translate_json(payload: dict, model=MODEL, max_retries=3) -> dict:
    prompt = PROMPT.format(json_payload=json.dumps(payload, ensure_ascii=False))

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=4096,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )

            raw = resp.choices[0].message.content.strip()

            try:
                return json.loads(raw)

            except json.JSONDecodeError as je:
                print("⚠️ JSON decode failed. Attempting salvage...")
                print(raw[:500])
                print("----")

                # 🔧 Salvage partial JSON like {"Article": "TEXT...
                key = list(payload.keys())[0]

                match = re.search(rf'"{re.escape(key)}"\s*:\s*"(.+)$', raw, re.DOTALL)
                if match:
                    salvaged = match.group(1)
                    print("🛠️ Salvaged truncated output")
                    return {key: salvaged}

                last_err = je
                time.sleep(2 ** attempt)
                continue

        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)

    raise RuntimeError(f"translate_json failed after {max_retries} retries: {last_err}")



def checkpoint_save(rows, path=OUT_PATH):
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------- Main ------------------
df = pd.read_csv(IN_PATH)

# Resume logic
if os.path.exists(OUT_PATH):
    existing = pd.read_csv(OUT_PATH)
    out_rows = existing.to_dict("records")
    start_idx = len(out_rows)
    print(f"🔁 Resuming from row {start_idx}")
else:
    out_rows = []
    start_idx = 0
    print("🆕 Starting fresh")

# Rebuild caches from saved output
article_cache = {}
summary_cache = {}

for r in out_rows:
    heb_a = clean_value(r.get("heb_article"))
    heb_s = clean_value(r.get("heb_summary"))
    if heb_a:
        article_cache[heb_a] = r.get("Article")
    if heb_s:
        summary_cache[heb_s] = r.get("Summary")

print(f"Cache rebuilt: {len(article_cache)} Articles, {len(summary_cache)} Summaries")

# ---------------- Processing Loop ----------------
for idx in range(start_idx, len(df)):
    try:
        row = df.iloc[idx]

        heb_article = clean_value(row.get("Article"))
        heb_summary = clean_value(row.get("Summary"))

        payload_other = {k: clean_value(row.get(k)) for k in FIELDS_OTHER}
        translated_other = {}
        for k, v in payload_other.items():
            translated_other[k] = safe_translate_field(k, v)

        if not heb_article and not heb_summary:
            print(f"⚠️ Skipping empty row {idx}")
            continue

        if heb_article:
            if heb_article not in article_cache:
                article_cache[heb_article] = translate_json({"Article": heb_article})["Article"]

        if heb_summary:
            if heb_summary not in summary_cache:
                summary_cache[heb_summary] = translate_json({"Summary": heb_summary})["Summary"]


        out_row = {
            "heb_article": heb_article,
            "heb_summary": heb_summary,
            "Article": article_cache.get(heb_article),
            "Summary": summary_cache.get(heb_summary),
            **translated_other,
        }

        out_rows.append(out_row)

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            checkpoint_save(out_rows)
            print(f"💾 Saved checkpoint at row {idx + 1}")

    except Exception as e:
        # 🔴 SAVE IMMEDIATELY ON ERROR
        print(f"\n❌ ERROR at row {idx}")
        print(str(e))
        print("💾 Saving progress before exit...")

        checkpoint_save(out_rows)

        print("🛑 Exiting. Rerun the script to resume.")
        sys.exit(1)

# ---------------- Final Save ----------------
checkpoint_save(out_rows)
pd.DataFrame(out_rows).drop(columns=["heb_article", "heb_summary"], errors="ignore") \
    .to_csv(CLEAN_OUT_PATH, index=False)

print(f"✅ Finished all rows ({len(out_rows)})")
print(f"🧹 Clean output written to {CLEAN_OUT_PATH}")
