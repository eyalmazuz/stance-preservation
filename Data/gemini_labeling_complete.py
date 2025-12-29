import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import json
import pandas as pd
import re
import ast
import time
from tqdm import tqdm
import google.generativeai as genai

# Initialize Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def clean_article_sentence(cell):
    """Clean the article sentence from tuple format if needed"""
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], tuple):
            return parsed[0][0]  # just return the sentence
    except:
        pass
    return cell

def clean_sentence_for_prompt(sentence):
    """Clean sentence for use in prompt"""
    if pd.isna(sentence) or sentence is None:
        return ""
    
    sentence = str(sentence)
    # Remove problematic characters and formatting
    sentence = sentence.replace('\\', '')
    sentence = sentence.replace('"', "'")
    sentence = sentence.replace('\n', ' ')
    sentence = sentence.strip()
    return sentence

def is_missing_value(value):
    """Check if value is missing (NaN, None, empty string, etc.)"""
    if pd.isna(value):
        return True
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False

def extract_labels_from_response(response, label_type="summary"):
    """Extract topic and stance from Gemini response"""
    try:
        # Try to find JSON block first
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1).strip()
            # Fix common JSON formatting issues
            json_str = re.sub(r'(?<=\n|{)\s*([a-zA-Z_][a-zA-Z0-9_\s]*?)\s*:', r'"\1":', json_str)
            json_str = json_str.replace("\\'", "'")
            json_str = re.sub(r',\s*}}', '}', json_str)
            
            try:
                data = json.loads(json_str)
                topic_key = f"{label_type} topic gemini"
                stance_key = f"{label_type} stance gemini"
                
                topic = data.get(topic_key, "").strip()
                stance = data.get(stance_key, "").strip()
                
                if topic and stance:
                    return topic, stance
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract from text using patterns
        topic_match = re.search(rf'{label_type} topic gemini["\s]*:[\s"]*([^"}}\n,]+)', response, re.IGNORECASE)
        stance_match = re.search(rf'{label_type} stance gemini["\s]*:[\s"]*([^"}}\n,]+)', response, re.IGNORECASE)
        
        if topic_match and stance_match:
            topic = topic_match.group(1).strip().strip('"').strip("'")
            stance = stance_match.group(1).strip().strip('"').strip("'")
            return topic, stance
    
    except Exception as e:
        print(f"Error extracting labels: {e}")
    
    return None, None

def get_labels_for_sentence(sentence, label_type="summary", max_retries=5):
    """Get topic and stance labels for a sentence using Gemini"""
    sentence_clean = clean_sentence_for_prompt(sentence)
    
    if not sentence_clean:
        return None, None
    
    prompt = f"""
עבור המשפט הבא: "{sentence_clean}"

קבע מה הנושא שלו (בקצרה, בין מילה לארבע מילים אם צריך)
לאחר מכן בדוק מה העמדה של המשפט הזה כלפי הנושא הזה.
האם הוא תומך בנושא, מתנגד לו או נייטרלי כלפיו?

זכור כי *עמדה* היא *לא* סנטימנט.
כלומר, אם הסנטימנט הוא שלילי זה לא אומר שהעמדה היא נגד. או אם הסנטימנט הוא חיובי זה לא אומר שהעמדה היא בעד.
העמדות האפשריות: [בעד, נגד, נייטרלי]

### Output Format:
```json
{{
    "{label_type} topic gemini": "your topic here",
    "{label_type} stance gemini": "your stance here"
}}
```
"""

    for attempt in range(max_retries):
        try:
            # Initialize the model
            model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
            
            # Generate response
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=500,
                )
            )
            
            if response.text:
                content = response.text
                topic, stance = extract_labels_from_response(content, label_type)
                
                if topic and stance:
                    return topic, stance
                
                print(f"Failed to extract labels from response (attempt {attempt + 1}): {content[:200]}...")
            else:
                print(f"Empty response from Gemini (attempt {attempt + 1})")
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"Gemini API error on attempt {attempt + 1}: {e}")
            
            # Handle specific quota/rate limit errors
            if "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                wait_time = 60 * (2 ** attempt)  # Wait longer for quota issues
                print(f"Quota/rate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            elif "500" in error_msg or "503" in error_msg:
                wait_time = 10 * (2 ** attempt)  # Server errors
                print(f"Server error. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Standard exponential backoff
    
    return None, None

def fill_missing_labels(csv_path, output_path=None, batch_size=10, delay=1):
    """Fill missing labels in the CSV file"""
    
    if output_path is None:
        output_path = csv_path.replace('.csv', '_filled.csv')
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Clean article sentences
    if 'Best Match Sentences From Article' in df.columns:
        df['Best Match Sentences From Article'] = df['Best Match Sentences From Article'].apply(clean_article_sentence)
    
    # Initialize columns if they don't exist
    required_columns = [
        'summary topic gemini', 'summary stance gemini',
        'article topic gemini', 'article stance gemini'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
    
    # Find rows that need processing
    summary_missing = []
    article_missing = []
    
    for idx, row in df.iterrows():
        # Check summary labels
        if (is_missing_value(row.get('summary topic gemini')) or 
            is_missing_value(row.get('summary stance gemini'))):
            if not is_missing_value(row.get('Sentence in Summary')):
                summary_missing.append(idx)
        
        # Check article labels
        if (is_missing_value(row.get('article topic gemini')) or 
            is_missing_value(row.get('article stance gemini'))):
            if not is_missing_value(row.get('Best Match Sentences From Article')):
                article_missing.append(idx)
    
    print(f"Found {len(summary_missing)} rows missing summary labels")
    print(f"Found {len(article_missing)} rows missing article labels")
    
    total_to_process = len(summary_missing) + len(article_missing)
    if total_to_process == 0:
        print("No missing labels found!")
        return df
    
    print(f"Total API calls needed: {total_to_process}")
    # Gemini Flash is much cheaper than OpenAI
    estimated_cost = total_to_process * 0.0001  # Very rough estimate for Gemini Flash
    print(f"Estimated cost: ~${estimated_cost:.4f} (Gemini is very cheap!)")
    
    proceed = input("Continue? (y/n): ")
    if proceed.lower() != 'y':
        return df
    
    processed = 0
    
    # Process summary labels
    if summary_missing:
        print(f"\nProcessing {len(summary_missing)} summary labels...")
        for idx in tqdm(summary_missing, desc="Summary labels"):
            sentence = df.at[idx, 'Sentence in Summary']
            topic, stance = get_labels_for_sentence(sentence, "summary")
            
            if topic and stance:
                df.at[idx, 'summary topic gemini'] = topic
                df.at[idx, 'summary stance gemini'] = stance
                processed += 1
            else:
                print(f"Failed to get labels for summary row {idx}")
            
            # Save progress periodically and add extra delay for quota management
            if processed % batch_size == 0:
                df.to_csv(output_path, index=False)
                print(f"Progress saved. Processed {processed}/{total_to_process}")
                # Extra pause after batch save to be quota-friendly
                time.sleep(2)
            
            time.sleep(delay)  # Rate limiting
    
    # Process article labels
    if article_missing:
        print(f"\nProcessing {len(article_missing)} article labels...")
        for idx in tqdm(article_missing, desc="Article labels"):
            sentence = df.at[idx, 'Best Match Sentences From Article']
            topic, stance = get_labels_for_sentence(sentence, "article")
            
            if topic and stance:
                df.at[idx, 'article topic gemini'] = topic
                df.at[idx, 'article stance gemini'] = stance
                processed += 1
            else:
                print(f"Failed to get labels for article row {idx}")
            
            # Save progress periodically and add extra delay for quota management
            if processed % batch_size == 0:
                df.to_csv(output_path, index=False)
                print(f"Progress saved. Processed {processed}/{total_to_process}")
                # Extra pause after batch save to be quota-friendly
                time.sleep(2)
            
            time.sleep(delay)  # Rate limiting
    
    # Final save
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Processing complete!")
    print(f"Successfully filled {processed}/{total_to_process} missing labels")
    print(f"Output saved to: {output_path}")
    
    # Summary statistics
    summary_filled = len(df[~df['summary topic gemini'].isna() & (df['summary topic gemini'].str.strip() != "")])
    article_filled = len(df[~df['article topic gemini'].isna() & (df['article topic gemini'].str.strip() != "")])
    
    print(f"\nFinal statistics:")
    print(f"Rows with summary labels: {summary_filled}/{len(df)}")
    print(f"Rows with article labels: {article_filled}/{len(df)}")
    
    return df

if __name__ == "__main__":
    # Configuration
    CSV_PATH = "./Data/datasets/labeled_data_final_complete.csv"  
    OUTPUT_PATH = "./Data/datasets/labeled_data_final_complete_2.csv"  
    
    # Parameters
    BATCH_SIZE = 10  # Save progress every N processed items
    DELAY = 1.0  # Delay between API calls (seconds) - Conservative for quota management
    MAX_RETRIES = 5  # Increased retries for quota issues
    
    # Test Gemini connection
    try:
        test_model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
        test_response = test_model.generate_content("Test")
        print("✅ Gemini API connection successful!")
    except Exception as e:
        print(f"❌ Gemini API connection failed: {e}")
        print("Please check your GEMINI_API_KEY in the .env file")
        exit(1)
    
    # Run the filling process
    df_filled = fill_missing_labels(
        csv_path=CSV_PATH,
        output_path=OUTPUT_PATH,
        batch_size=BATCH_SIZE,
        delay=DELAY
    )