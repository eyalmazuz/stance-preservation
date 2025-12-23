import json
import random
from typing import Any
import re
import os
from datetime import datetime
import csv
import pandas as pd
from datasets import Dataset
from collections import namedtuple
import torch
import torch.nn.functional as F
import numpy as np
from transformers import TrainingArguments, Trainer
from evaluate import load
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import ot
from math import log as ln
from torch.distributions import Categorical
from sentence_transformers import SentenceTransformer, util
from typing import Optional
import gc
from peft import PeftModel

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM
    )


# import nltk
# nltk.download('punkt_tab')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IDX2SOURCE = {
    0: "Weizmann",
    1: "Wikipedia",
    2: "Bagatz",
    3: "Knesset",
    4: "Israel_Hayom",
}


def load_data(path: str) -> list[dict[str, Any]]:
    with open(path) as fd:
        summaries = [json.loads(line) for line in fd.readlines()]

    return summaries


def get_train_test_split(
    summaries: list[dict[str, Any]],
    split_type: str,
    source_type: str,
    test_size: Optional[float] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split_type.lower() == "random":
        if test_size is not None:
            random.shuffle(summaries)
            train_set = summaries[int(len(summaries) * test_size) :]
            test_set = summaries[: int(len(summaries) * test_size)]
        else:
            raise ValueError("Test size can't be None")

    elif split_type.lower() == "source":
        train_set = [summary for summary in summaries if summary["metadata"]["source"] != source_type]

        test_set = [summary for summary in summaries if summary["metadata"]["source"] == source_type]

    else:
        raise ValueError(f"Invlid split type was selected {split_type}")

    return train_set, test_set


def extract_texts(
    summaries: list[dict[str, Any]],
    only_summaries: bool,
) -> list[str]:
    positives: list[str] = []
    for summary in summaries:
        if (
            not only_summaries
            and "text_raw" in summary
            and summary["text_raw"] is not None
            and summary["text_raw"] != ""
        ):
            positives.append(summary["text_raw"])

        if (
            "ai_summary" in summary["metadata"]
            and summary["metadata"]["ai_summary"] is not None
            and summary["metadata"]["ai_summary"] != ""
        ):
            positives.append(summary["metadata"]["ai_summary"])

        if "summary" in summary and summary["summary"] is not None and summary["summary"] != "":
            positives.append(summary["summary"])

    return list(set(positives))


# --------------------------------------------------------------- sentence matching functions --------------------------------------------------------------- 
def split_into_sentences(text):
    """Split text into sentences."""
    if not isinstance(text, str):
        return []
    separators = r"[■|•.\n]"
    sentences = [sent.strip() for sent in re.split(separators, text) if sent.strip()]
    return sentences

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    output_dir = "./Data/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def generate_output_filename(dataset_name, file_type="results"):
    """Generate standardized output filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean up dataset name by removing any path components and special characters
    dataset_name = os.path.basename(dataset_name).replace('/', '_')
    return f"{dataset_name}_{file_type}_{timestamp}.csv"

def save_results(results_data, dataset_name, file_type="results"):
    """Save results to CSV in the output directory."""
    output_dir = ensure_output_dir()
    filename = generate_output_filename(dataset_name, file_type)
    output_path = os.path.join(output_dir, filename)

    try:
        print(f"Saving results to {output_path}...")
        if results_data:
            # Dynamically generate fieldnames from the keys of the first entry
            fieldnames = results_data[0].keys()

            with open(output_path, mode="w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_data)
        print("Results saved successfully.")
        return output_path
    except Exception as e:
        print(f"Error saving results: {e}")
        return None

def process_dataset_item(dataset, idx):
    """Process both HuggingFace and custom dataset formats."""
    if isinstance(dataset, Dataset):
        # For HuggingFace datasets
        article = dataset[idx]['article']
        summary = dataset[idx]['summary']
    elif isinstance(dataset, dict):
        # For dictionary-like datasets
        article = dataset['article'][idx]
        summary = dataset['summary'][idx]
    else:
        raise ValueError(f"Unexpected dataset format: {type(dataset)}")
    return article, summary


def get_dataset_length(dataset):
    """Get length of dataset regardless of its format."""
    if isinstance(dataset, Dataset):
        return len(dataset)
    elif isinstance(dataset, dict):
        return len(dataset['article'])
    else:
        raise ValueError(f"Unexpected dataset format: {type(dataset)}")

# def process_and_display_results(dataset, match_fn, dataset_name, save_matches=False, threshold=0.8, top_k_matches=1):
#     """Process and display results for article matching and save to CSV."""
#     results_data = []
#     metadata_data = []  

#     # define namedtuple
#     # Match = namedtuple('Match', ['summary_sentence', 'article_sentences'])
#     Match = namedtuple('Match', [
#         'summary_sentence', 
#         'article_sentences',
#         'emd',
#         'emd_mean',
#         'kl_divergences',
#         'kl_mean',
#         'summary_stance',
#         'summary_stance_score',
#         'article_stances',
#         'article_stance_scores'
#     ])
#     matches = []

#     dataset_length = get_dataset_length(dataset)
#     # num_articles_to_process = min(num_articles, dataset_length)

#     # print("\nArticle Matching Results")
#     # print(f"Processing {num_articles_to_process} articles out of {dataset_length} total articles")

#     # for idx in range(dataset_length):
#     for idx in range(451):
#         # print(f"\nArticle {idx + 1}")

#         try:
#             article, summary = process_dataset_item(dataset, idx)

#             source_chunks = split_into_sentences(article)
#             target_chunks = split_into_sentences(summary)

#             # source_chunks = smart_split_by_similarity(article)
#             # target_chunks = smart_split_by_similarity(summary)

#             if not source_chunks or not target_chunks:
#                 print(f"Warning: Empty chunks found for article {idx + 1}, skipping...")
#                 continue

#             results = match_fn(source_chunks, target_chunks)
#             source, target, matching_matrix = results

#             # matching_df = pd.DataFrame(
#             #     matching_matrix,
#             #     index=[f"Target {i + 1}" for i in range(len(target_chunks))],
#             #     columns=[f"Source {j + 1}" for j in range(len(source_chunks))]
#             # )
#             # print("Matching Matrix:")
#             # print(matching_df)

#             # Prepare results data
#             for i, target_sentence in enumerate(target_chunks):
#                 best_matches = []
#                 # best_matches_scores = []
#                 for j, source_sentence in enumerate(source_chunks):
#                     if matching_matrix[i, j] >= threshold:
#                         best_matches.append((source_sentence, matching_matrix[i, j]))
#                         # best_matches_scores.append(matching_matrix[i, j])
#                 best_matches = sorted(best_matches, key=lambda x: x[1], reverse=True)[:top_k_matches]
#                 match = Match(
#                     summary_sentence=target_sentence, 
#                     article_sentences=best_matches,
#                     emd=None,
#                     emd_mean=None,
#                     kl_divergences=None,       # Default to None
#                     kl_mean=None,              # Default to None
#                     summary_stance=None,       # Default to None
#                     summary_stance_score=None, # Default to None
#                     article_stances=None,      # Default to None
#                     article_stance_scores=None # Default to None
#                 )
                
#                 matches.append(match)
#                 results_data.append({
#                     "Article": article,
#                     "Summary": summary,
#                     "Sentence in Summary": target_sentence,
#                     "Best Match Sentences From Article": best_matches,
#                     # "Best Match Score": best_matches_scores,
#                 })

#             # Prepare metadata
#             metadata_data.append({
#                 "Article_ID": idx,
#                 "Num_Source_Sentences": len(source_chunks),
#                 "Num_Target_Sentences": len(target_chunks),
#                 "Average_Match_Score": matching_matrix.mean(),
#                 "Max_Match_Score": matching_matrix.max(),
#                 "Min_Match_Score": matching_matrix.min()
#             })

#         except Exception as e:
#             print(f"Error processing Article {idx + 1}: {str(e)}")
#             continue

#     # Save results and metadata
#     # if save_matches and results_data:
#     #     save_results(results_data, dataset_name, "results")
#     #     save_results(metadata_data, dataset_name, "metadata")

#     return matches, results_data


def process_and_display_results(dataset, match_fn, dataset_name, save_matches=False, threshold=0.8, top_k_matches=1, chunk_size=7):
    """Process and display results for article matching and save to CSV."""
    results_data = []
    metadata_data = []
    # length = 50
    length = get_dataset_length(dataset)
    num_chunks = (length + chunk_size - 1) // chunk_size  

    # define namedtuple
    Match = namedtuple('Match', [
        'summary_sentence', 
        'article_sentences',
        'emd',
        'emd_mean',
        'kl_divergences',
        'kl_mean',
        'summary_stance',
        'summary_stance_score',
        'article_stances',
        'article_stance_scores'
    ])
    matches = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, length)
        print(f"Processing articles {start_idx + 1} to {end_idx}")

        for idx in range(start_idx, end_idx):
            try:
                article, summary = process_dataset_item(dataset, idx)
                source_chunks = split_into_sentences(article)
                target_chunks = split_into_sentences(summary)

                if not source_chunks or not target_chunks:
                    print(f"Warning: Empty chunks found for article {idx + 1}, skipping...")
                    continue

                results = match_fn(source_chunks, target_chunks)
                source, target, matching_matrix = results

                # Prepare results data
                for i, target_sentence in enumerate(target_chunks):
                    best_matches = []
                    for j, source_sentence in enumerate(source_chunks):
                        if matching_matrix[i, j] >= threshold:
                            best_matches.append((source_sentence, matching_matrix[i, j]))
                    best_matches = sorted(best_matches, key=lambda x: x[1], reverse=True)[:top_k_matches]
                    match = Match(
                        summary_sentence=target_sentence, 
                        article_sentences=best_matches,
                        emd=None,
                        emd_mean=None,
                        kl_divergences=None,
                        kl_mean=None,
                        summary_stance=None,
                        summary_stance_score=None,
                        article_stances=None,
                        article_stance_scores=None
                    )
                    
                    matches.append(match)
                    results_data.append({
                        "Article": article,
                        "Summary": summary,
                        "Sentence in Summary": target_sentence,
                        "Best Match Sentences From Article": best_matches,
                    })

                # Prepare metadata
                metadata_data.append({
                    "Article_ID": idx,
                    "Num_Source_Sentences": len(source_chunks),
                    "Num_Target_Sentences": len(target_chunks),
                    "Average_Match_Score": matching_matrix.mean(),
                    "Max_Match_Score": matching_matrix.max(),
                    "Min_Match_Score": matching_matrix.min()
                })

            except Exception as e:
                print(f"Error processing Article {idx + 1}: {str(e)}")
                continue

        # Save intermediate results after each chunk
        if save_matches and results_data:
            save_results(results_data, dataset_name, f"results_chunk_{chunk_idx}")
            save_results(metadata_data, dataset_name, f"metadata_chunk_{chunk_idx}")

        # Clear GPU cache and Python garbage
        torch.cuda.empty_cache()
        gc.collect()

    return matches, results_data



# --------------------------------------------------------------- pipeline functions ---------------------------------------------------------------
def load_matching_matrix(csv_path):
    """Load the sentence matching matrix from a CSV file."""
    df = pd.read_csv(csv_path)
    print("Matching Matrix Loaded:")
    print(df.head())
    return df

def classify_stance(sentences, model, tokenizer):
    """Classify stance (sentiment) for a list of sentences and return labels with probabilities."""
    labels = ['דת ומדינה_תומך', 'דת ומדינה_מתנגד', 'דת ומדינה_נייטרלי', 
              'פוליטיקה וממשל_תומך', 'פוליטיקה וממשל_מתנגד', 'פוליטיקה וממשל_נייטרלי', 
              'הסכסוך הישראלי-פלסטיני_תומך', 'הסכסוך הישראלי-פלסטיני_מתנגד', 'הסכסוך הישראלי-פלסטיני_נייטרלי', 
              'ביטחון_תומך', 'ביטחון_מתנגד', 'ביטחון_נייטרלי', 
              'כלכלה וחברה_תומך', 'כלכלה וחברה_מתנגד', 'כלכלה וחברה_נייטרלי', 
              'זכויות אדם ושוויון_תומך', 'זכויות אדם ושוויון_מתנגד', 'זכויות אדם ושוויון_נייטרלי', 
              'תקשורת ודמוקרטיה_תומך', 'תקשורת ודמוקרטיה_מתנגד', 'תקשורת ודמוקרטיה_נייטרלי', 
              'בריאות ורווחה_תומך', 'בריאות ורווחה_מתנגד', 'בריאות ורווחה_נייטרלי', 
              'חינוך ותרבות_תומך', 'חינוך ותרבות_מתנגד', 'חינוך ותרבות_נייטרלי', 
              'תחבורה ותשתיות_תומך', 'תחבורה ותשתיות_מתנגד', 'תחבורה ותשתיות_נייטרלי', 
              'סביבה ואקלים_תומך', 'סביבה ואקלים_מתנגד', 'סביבה ואקלים_נייטרלי', 
              'מדיניות חוץ והגירה_תומך', 'מדיניות חוץ והגירה_מתנגד', 'מדיניות חוץ והגירה_נייטרלי', 
              'פיתוח אזורי_תומך', 'פיתוח אזורי_מתנגד', 'פיתוח אזורי_נייטרלי', 
              'אלימות וחוק_תומך', 'אלימות וחוק_מתנגד', 'אלימות וחוק_נייטרלי', 
              'חדשנות וטכנולוגיה_תומך', 'חדשנות וטכנולוגיה_מתנגד', 'חדשנות וטכנולוגיה_נייטרלי', 
              'ספורט_תומך', 'ספורט_מתנגד', 'ספורט_נייטרלי']
    results = []
    probs = []
    
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()
        predicted_class = torch.argmax(logits, dim=1).item()
        results.append((labels[predicted_class], probabilities[predicted_class]))
        probs.append(probabilities)
    
    return results, probs

def kl_divergence(p, q):
    """Compute KL-divergence between two probability distributions."""
    p = torch.tensor(p)
    q = torch.tensor(q)
    return F.kl_div(q.log(), p, reduction='sum').item()

def compute_stance_preservation(data, model, tokenizer):
    """Compute stance preservation and KL-divergence between sentiment distributions."""
    updated_data = []
    
    for match in data:
        # Summary sentence
        summary_sentence = match.summary_sentence
        
        # Classify stance for summary sentence
        summary_results, summary_probs = classify_stance([summary_sentence], model, tokenizer)
        summary_stance, summary_score = summary_results[0]
        
        # Process each article sentence
        article_stances = []
        article_stance_scores = []
        article_probs_list = []
        
        for article_sentence in match.article_sentences:
            # Classify stance for each article sentence
            article_results, article_probs = classify_stance([article_sentence[0]], model, tokenizer)
            article_stance, article_score = article_results[0]
            
            article_stances.append(article_stance)
            article_stance_scores.append(article_score)
            article_probs_list.append(article_probs[0])
        
        # Compute KL divergences for each article sentence
        kl_scores = [kl_divergence(summary_probs[0], article_prob) 
                     for article_prob in article_probs_list]
        # Compute mean KL divergence
        kl_mean = np.mean(kl_scores)

        # Create a new namedtuple with additional information
        updated_match = match._replace(
            kl_divergences=kl_scores,
            kl_mean=kl_mean,
            summary_stance=summary_stance,
            summary_stance_score=summary_score,
            article_stances=article_stances,
            article_stance_scores=article_stance_scores
        )
        
        updated_data.append(updated_match)
    
    return updated_data

def get_topic_stance_values(class_index, num_stances_per_topic=3):
    topic_id = class_index // num_stances_per_topic
    stance_value = class_index % num_stances_per_topic  # 0, 1, or 2
    return topic_id, stance_value

def compute_stance_preservation_emd(data, model, tokenizer):
    """Compute stance preservation and emd between sentiment distributions."""
    updated_data = []
    Topic_Mismatch_Penalty = 3.0
    num_labels = 48  
    num_classes = 48


    
    for match in data:
        # Summary sentence
        summary_sentence = match.summary_sentence
        
        # Classify stance for summary sentence
        summary_results, summary_probs = classify_stance([summary_sentence], model, tokenizer)
        summary_stance, summary_score = summary_results[0]
        
        # Process each article sentence
        article_stances = []
        article_stance_scores = []
        article_probs_list = []
        
        for article_sentence in match.article_sentences:
            # Classify stance for each article sentence
            article_results, article_probs = classify_stance([article_sentence[0]], model, tokenizer)
            article_stance, article_score = article_results[0]
            
            article_stances.append(article_stance)
            article_stance_scores.append(article_score)
            article_probs_list.append(article_probs[0])
        
        distance_matrix = np.zeros((num_classes, num_classes))
        
        for i in range(num_classes):
            topic_i, stance_val_i = get_topic_stance_values(i)
            for j in range(num_classes):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    topic_j, stance_val_j = get_topic_stance_values(j)
                    if topic_i == topic_j:
                        distance_matrix[i, j] = float(abs(stance_val_i - stance_val_j))
                    else:
                        distance_matrix[i, j] = Topic_Mismatch_Penalty

        
        summary_probs[0] /= np.sum(summary_probs[0])
        summary_tensor = torch.tensor(summary_probs[0])

        emd_scores = []
        for article_prob in article_probs_list:
            article_tensor = torch.tensor(article_prob)
            sum_entropy = Categorical(probs=summary_tensor).entropy()
            art_entropy = Categorical(probs=article_tensor).entropy()
            if sum_entropy > ln(num_classes) / 2 or art_entropy > ln(num_classes) / 2:
                # don't include the pair in the final metric value 
                continue

            article_prob /= np.sum(article_prob)

            model_probabilities_f64 = np.asarray(summary_probs[0], dtype=np.float64)
            target_probabilities_f64 = np.asarray(article_prob, dtype=np.float64)
            distance_matrix_f64 = np.asarray(distance_matrix, dtype=np.float64)

            emd_value = ot.emd2(model_probabilities_f64, target_probabilities_f64, distance_matrix_f64)
            emd_scores.append(emd_value)

        if len(emd_scores) > 0:
            emd_mean = np.mean(emd_scores)
        else:
            emd_mean = None            
            
        # Create a new namedtuple with additional information
        updated_match = match._replace(
            emd=emd_scores,
            emd_mean=emd_mean,
            summary_stance=summary_stance,
            summary_stance_score=summary_score,
            article_stances=article_stances,
            article_stance_scores=article_stance_scores
        )
        
        updated_data.append(updated_match)
    
    return updated_data

# def preprocess(example, tokenizer, label2id):
#     combined = f"{example['sentence']} [SEP] {example['topic']}"
#     inputs = tokenizer(
#         combined,
#         truncation=True,
#         padding="max_length",
#         max_length=128
#     )
#     inputs["label"] = label2id[example["stance"]]
#     return inputs


def classify_stance_with_topic(sentence, topic, model, tokenizer):
    """Classify stance for a single sentence with topic context."""
    # Prepare labels for Hebrew stance classification
    labels = ['בעד', 'נגד', 'נייטרלי']  # favor, against, neutral
    
    # Combine sentence with topic
    combined_input = f"{sentence} [SEP] {topic}"
    
    inputs = tokenizer(combined_input, return_tensors='pt', truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1).squeeze().tolist()
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return labels[predicted_class], probabilities[predicted_class], probabilities


def compute_stance_preservation_with_topic(dataset, model, tokenizer):
    """Compute stance preservation between article and summary sentences."""
    results = []
    Topic_Mismatch_Penalty = 3.0
    num_classes = 3

    for item in dataset:
        try:
            # Get sentences and topics
            article_sentence = item["article_sentence"]
            article_topic = item["article_topic"]
            summary_sentence = item["summary_sentence"]
            summary_topic = item["summary_topic"]
            
            # Classify stance for both sentences
            summary_stance, summary_score, summary_probs = classify_stance_with_topic(
                summary_sentence, summary_topic, model, tokenizer
            )
            
            article_stance, article_score, article_probs = classify_stance_with_topic(
                article_sentence, article_topic, model, tokenizer
            )
            
            # Create distance matrix for EMD calculation
            distance_matrix = np.zeros((num_classes, num_classes))
            
            for i in range(num_classes):
                topic_i, stance_val_i = get_topic_stance_values(i)
                for j in range(num_classes):
                    if i == j:
                        distance_matrix[i, j] = 0.0
                    else:
                        topic_j, stance_val_j = get_topic_stance_values(j)
                        if topic_i == topic_j:
                            distance_matrix[i, j] = float(abs(stance_val_i - stance_val_j))
                        else:
                            distance_matrix[i, j] = Topic_Mismatch_Penalty
            
            # Normalize probabilities
            summary_probs = np.array(summary_probs) / np.sum(summary_probs)
            article_probs = np.array(article_probs) / np.sum(article_probs)
            
            # Calculate entropy to filter out uncertain predictions
            summary_tensor = torch.tensor(summary_probs)
            article_tensor = torch.tensor(article_probs)
            
            sum_entropy = Categorical(probs=summary_tensor).entropy()
            art_entropy = Categorical(probs=article_tensor).entropy()
            
            # Skip if entropy is too high (uncertain predictions)
            if sum_entropy > ln(num_classes) / 2 or art_entropy > ln(num_classes) / 2:
                continue
            
            # Calculate EMD (Earth Mover's Distance)
            model_probabilities_f64 = np.asarray(summary_probs, dtype=np.float64)
            target_probabilities_f64 = np.asarray(article_probs, dtype=np.float64)
            distance_matrix_f64 = np.asarray(distance_matrix, dtype=np.float64)
            
            emd_value = ot.emd2(model_probabilities_f64, target_probabilities_f64, distance_matrix_f64)
            
            # Store results
            result = {
                "article_sentence": article_sentence,
                "article_topic": article_topic,
                "article_stance": article_stance,
                "article_stance_score": article_score,
                "article_probabilities": article_probs.tolist(),
                "summary_sentence": summary_sentence,
                "summary_topic": summary_topic,
                "summary_stance": summary_stance,
                "summary_stance_score": summary_score,
                "summary_probabilities": summary_probs.tolist(),
                "emd_score": emd_value,
                "stance_match": article_stance == summary_stance
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    return results


def load_topic_model(model_name):
    print("Setting up topic detection model...")

    quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  
        ) 
    
    
    quant_config_8 = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,          
        llm_int8_has_fp16_weight=False    
    )
    
    if model_name == 'dicta-il/dictalm2.0':
        # Topic detection setup
        topic_model = AutoModelForCausalLM.from_pretrained('dicta-il/dictalm2.0', device_map='cuda', quantization_config=quant_config)
        topic_tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictalm2.0')
        
        # Fix tokenizer configuration
        if topic_tokenizer.pad_token is None:
            topic_tokenizer.pad_token = topic_tokenizer.eos_token

    
    elif model_name == 'finetuned':
        # Fine-tuned dicta model for topic detection
        base_model_name = "dicta-il/dictalm2.0"

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="cuda",
            quantization_config=quant_config
        )

        topic_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        if topic_tokenizer.pad_token is None:
            topic_tokenizer.pad_token = topic_tokenizer.eos_token

        lora_model_path = "./models/results_topic_detection/final_adapters/"

        topic_model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            device_map="auto"
        )

        topic_model.eval()

    
    else:
        # model_name = "google/gemma-2-9b"
        topic_tokenizer = AutoTokenizer.from_pretrained(model_name)
        topic_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config_8,
        )

    return topic_model, topic_tokenizer

    

# --------------------------------------------------------------- finetuning functions ---------------------------------------------------------------
# Dataset class
class StanceDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_length=512):
        self.texts = texts
        self.labels = [label2id[label] for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Handle both single index and list of indices
        if isinstance(idx, int):
            # Single index case
            text = self.texts[idx]
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Remove batch dimension added by tokenizer when return_tensors="pt"
            encoding = {key: val.squeeze(0) for key, val in encoding.items()}
            encoding["labels"] = torch.tensor(label, dtype=torch.long)
            return encoding
        else:
            # This should not happen as DataLoader normally calls __getitem__ with integers
            # But we'll handle it by returning a list of individual items
            return [self.__getitem__(i) for i in idx]


# Custom Trainer with class weights
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  
        labels = inputs.get("labels")  
        
        # Forward pass with all inputs including labels
        outputs = model(**inputs)
        
        # Handle different output formats
        if hasattr(outputs, 'logits'):
            # Standard transformers model output
            raw_logits = outputs.logits
        elif isinstance(outputs, tuple) and len(outputs) >= 2:
            # Custom model returning (loss, logits)
            raw_logits = outputs[1]
        elif isinstance(outputs, tuple) and len(outputs) == 1:
            # Only logits returned
            raw_logits = outputs[0]
        else:
            # Direct logits tensor
            raw_logits = outputs

        # Cast these raw_logits to float32 for the weighted loss calculation
        logits_for_weighted_loss_calc = raw_logits.float()

        loss_fct_args = {}
        if self.class_weights is not None:
            current_device = logits_for_weighted_loss_calc.device
            weights = self.class_weights.to(device=current_device, dtype=torch.float32)
            loss_fct_args["weight"] = weights
        
        loss_fct = nn.CrossEntropyLoss(**loss_fct_args)
        
        # Calculate the weighted loss
        recalculated_loss = loss_fct(logits_for_weighted_loss_calc, labels)
        
        if return_outputs:
            # Return the outputs in the expected format
            if hasattr(outputs, 'logits'):
                # For standard transformers outputs, create a new output object
                from transformers.modeling_outputs import SequenceClassifierOutput
                new_outputs = SequenceClassifierOutput(
                    loss=recalculated_loss,
                    logits=raw_logits,
                    hidden_states=getattr(outputs, 'hidden_states', None),
                    attentions=getattr(outputs, 'attentions', None)
                )
                return (recalculated_loss, new_outputs)
            else:
                # For tuple outputs
                return (recalculated_loss, (recalculated_loss, raw_logits))
        else:
            return recalculated_loss


def load_model(model_name):
    """Load the stance detection model and tokenizer."""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Model and tokenizer loaded successfully from {model_name}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# Prediction
def predict_stance(text, model, tokenizer, id2label):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return id2label[prediction]


# Evaluation
def evaluate_model(model, eval_dataloader):
    model.to(device)
    model.eval()
    predictions, references = [], []

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        preds = torch.argmax(outputs.logits, dim=-1)
        predictions.extend(preds.cpu().tolist())
        references.extend(batch["labels"].cpu().tolist())

    metric = load("accuracy")
    return metric.compute(predictions=predictions, references=references)


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=-1).tolist()
#     f1 = f1_score(labels, preds, average='macro')
#     acc = accuracy_score(labels, preds)

#     accuracy = load("accuracy").compute(predictions=preds, references=labels)
#     f1_macro = load("f1").compute(predictions=preds, references=labels, average="macro")
#     return {
#         "f1_macro": f1,
#         "accuracy": acc,
#     }

def compute_metrics(p): 
    # Handle different prediction formats
    if hasattr(p, 'predictions'):
        predictions = p.predictions
    else:
        predictions = p
    
    # Extract logits from different possible formats
    if isinstance(predictions, tuple):
        # If predictions is a tuple, try to get logits
        if len(predictions) >= 2:
            logits_tensor = predictions[1]  # Usually (loss, logits)
        else:
            logits_tensor = predictions[0]
    else:
        logits_tensor = predictions
    
    # Convert to numpy if it's a tensor
    if hasattr(logits_tensor, 'detach'):
        logits_tensor = logits_tensor.detach().cpu().numpy()
    
    preds = np.argmax(logits_tensor, axis=1)
    
    # Handle labels
    if hasattr(p, 'label_ids'):
        labels = p.label_ids
    else:
        labels = p.labels
    
    # Convert labels to numpy if needed
    if hasattr(labels, 'detach'):
        labels = labels.detach().cpu().numpy()
    
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')  # Changed to macro for better handling of imbalanced classes
    f1_weighted = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': acc,
        'f1': f1_weighted,
        'f1_macro': f1_macro,
    }


