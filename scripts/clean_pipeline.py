import pandas as pd
from src.data_loader import load_data
from src.models.me5 import create_matching_matrix_with_e5_instruct
import argparse
from src.utils import (
    ensure_output_dir, 
    process_and_display_results, 
    compute_stance_preservation_with_topic,
    load_topic_model
)
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer
    )
import torch
from scripts.topic_detection import get_topic_for_model, get_topic_for_model_eng
import numpy as np


# python -m scripts.stance_pipeline --data custom --path Data/test_data.csv  --output-dir ./scripts/output/test_csv_input.json --save-matches
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run stance pipeline.")

    parser.add_argument(
        "--data", 
        type=str, 
        required=True, 
        help="Dataset name (e.g., biunlp/HeSum or custom or other text (for csv))."
        )
    
    parser.add_argument(
        "--path", 
        default="./Data/datasets/summarization-7-heb.jsonl", # ./Data/datasets/English_all_data_clean.csv
        type=str, 
        help="Path to custom dataset JSON."
        )
    
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.85, 
        help="Matching threshold."
        )
    
    parser.add_argument(
        "--top-k-matches", 
        type=int, 
        default=1, 
        help="Number of top matches to consider."
        )
    
    parser.add_argument(
        "--save-matches", 
        action="store_true", 
        help="Save matching results to CSV."
        )
    
    parser.add_argument(
        "--output-dir", 
        default="./scripts/output/stance_preservation_test.json", 
        help="Save stance preservation results to JSON."
        )
    
    parser.add_argument(
        "--topic_detection_model", 
        default="dicta-il/dictalm2.0", # google/gemma-2-9b
        help="dicta-il/dictalm2.0 or finetuned or huggingface model path (e.g. google/gemma-2-9b)"
        )
    
    parser.add_argument(
        "--language", 
        default="Hebrew", # English
        help="Language of the data (e.g. Hebrew or English)"
        )
    
    args = parser.parse_args()

    # data loading
    try:
        # Ensure output directory exists
        ensure_output_dir()

        # Load dataset
        dataset = load_data(args.data, args.path)
        print("Dataset loaded and preprocessed.")

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

    # process data and sentence matching
    try:
        match_fn = create_matching_matrix_with_e5_instruct

        # Process and display results, and save to CSV
        dataset_name = args.path if args.path else args.data
        data, results_data = process_and_display_results(dataset, match_fn, dataset_name, args.save_matches, args.threshold, args.top_k_matches)
        print(f"Processed {len(data)} matches")

        # save data as csv
        data_df = pd.DataFrame(results_data)
        data_df.to_csv("./Data/datasets/english_labeled_data.csv", index=False)

    except Exception as e:
        print(f"Error in sentence matching and data processing: {str(e)}")
        raise
    
    # Topic detection model setup
    try:
        topic_model, topic_tokenizer = load_topic_model(args.topic_detection_model)
    except Exception as e:
        print(f"Error loading topic detection model: {str(e)}")
        raise

    # Process sentences for topic detection
    try:
        topic_model_name = args.topic_detection_model
        topic_dataset = []
        total_sentences = len(data)
        
        for idx, sent in enumerate(data):
            # Get the topic for the current sentence
            if hasattr(sent, 'article_sentences') and sent.article_sentences:
                article_sent = sent.article_sentences[0][0]  # Get first sentence from first match
                summary_sent = sent.summary_sentence

                if args.language == "Hebrew":
                    art_res = get_topic_for_model(results_data[idx]["Article"], article_sent, topic_model, topic_tokenizer, topic_model_name)
                    sum_res = get_topic_for_model(results_data[idx]["Summary"], summary_sent, topic_model, topic_tokenizer, topic_model_name)
                else:  # English
                    art_res = get_topic_for_model_eng(results_data[idx]["Article"], article_sent, topic_model, topic_tokenizer, topic_model_name)
                    sum_res = get_topic_for_model_eng(results_data[idx]["Summary"], summary_sent, topic_model, topic_tokenizer, topic_model_name)

                # Check topic similarity
                if art_res["topic"] == sum_res["topic"]:
                    res = {
                        "article_sentence": art_res["sentence"],
                        "article_topic": art_res["topic"],
                        "summary_sentence": sum_res["sentence"],
                        "summary_topic": sum_res["topic"]
                    }
                    # print(f"Topic match found: {res['article_topic']}")
                    topic_dataset.append(res)

            # Print progress every 10 sentences
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{total_sentences} sentences")

        print(f"Found {len(topic_dataset)} sentence pairs with matching topics")

    except Exception as e:
        print(f"Error in topic detection processing: {str(e)}")
        raise

    if not topic_dataset:
        print("No matching topics found. Exiting.")
        exit(1)

    # Load stance detection model
    try:
        print("Loading stance detection model...")
        if args.language == "Hebrew":
            stance_model_name = './models/stance_detection_model_combined/'
            stance_model = AutoModelForSequenceClassification.from_pretrained(stance_model_name)
            stance_tokenizer = AutoTokenizer.from_pretrained(stance_model_name)
        else:  # English
            stance_model_name = 'bendavidsteel/Qwen3-1.7B-stance-detection'
            stance_model = AutoModelForSequenceClassification.from_pretrained(stance_model_name, num_labels=3)
            stance_tokenizer = AutoTokenizer.from_pretrained(stance_model_name)

    except Exception as e:
        print(f"Error loading stance detection model: {str(e)}")
        raise
    
    # Classify stance and compute preservation
    try:
        print("Computing stance preservation...")
        results = compute_stance_preservation_with_topic(topic_dataset, stance_model, stance_tokenizer, args.language)
        
        print(f"Stance preservation computed for {len(results)} sentence pairs")
        
    except Exception as e:
        print(f"Error in stance preservation computation: {str(e)}")
        raise

    # ------------------------------------------------------
    # Make the pipeline's output file like the labeled file
    # ------------------------------------------------------
    results_data_df = pd.DataFrame(results_data)
    results_df = pd.DataFrame(results)

    results_df = results_df.rename(columns={
        "article_sentence": "Best Match Sentences From Article",
        "summary_sentence": "Sentence in Summary" 
    })

    def extract_sentence(val):
    # If it's a list of tuples, take the first tuple's first element (the sentence)
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], tuple):
            return str(val[0][0])
        # If it's a list of strings, take the first string
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], str):
            return str(val[0])
        # Otherwise, just return as string
        return str(val)

    results_data_df["Best Match Sentences From Article"] = results_data_df["Best Match Sentences From Article"].apply(extract_sentence)
    results_data_df["Sentence in Summary"] = results_data_df["Sentence in Summary"].apply(str)
    results_df["Best Match Sentences From Article"] = results_df["Best Match Sentences From Article"].apply(str)
    results_df["Sentence in Summary"] = results_df["Sentence in Summary"].apply(str)

    # Now you can merge
    merged_df = pd.merge(
        results_data_df,
        results_df,
        on=["Best Match Sentences From Article", "Sentence in Summary"],
        how="inner"
    )
    # ------------------------------------------------------

    # Save results
    if args.save_matches and results:
        output_path = args.output_dir
        
        # Create DataFrame and save as JSON
        result_df = pd.DataFrame(results)
        result_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        print(f"Stance preservation results saved to {output_path}")

        # save the merged DataFrame to CSV
        merged_output_path = output_path.replace('.json', '_merged.csv')
        merged_df.to_csv(merged_output_path, index=False, encoding='utf-8')
        print(f"Merged results saved to {merged_output_path}")
        
        
        # Print summary statistics
        if len(results) > 0:
            stance_matches = sum(1 for r in results if r['stance_match'])
            avg_emd = np.mean([r['emd_score'] for r in results])
            print(f"Summary: {stance_matches}/{len(results)} stance matches, average EMD: {avg_emd:.4f}")
    else:
        print("No results to save or save_matches flag not set")
     

