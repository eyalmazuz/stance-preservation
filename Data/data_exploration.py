import json
import random
import pandas as pd
import os

def load_data(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def sample_sentences(data, min_samples=3, max_samples=5):
    """
    Sample random sentences from each topic and stance category.
    
    Args:
        data: List of dictionaries containing topic data
        min_samples: Minimum number of sentences to sample
        max_samples: Maximum number of sentences to sample
        
    Returns:
        Dictionary with sampled sentences organized by topic and sentiment
    """
    samples = {}
    
    for item in data:
        topic = item['topic']
        samples[topic] = {}
        
        # Sample from each sentiment category
        for sentiment in ['supportive_sentences', 'opposing_sentences', 'neutral_sentences']:
            display_name = sentiment.replace('_sentences', '')
            
            # Determine number of samples for this category
            num_sentences = len(item[sentiment])
            num_samples = min(random.randint(min_samples, max_samples), num_sentences)
            
            # Get random samples
            sampled_indices = random.sample(range(num_sentences), num_samples)
            sampled_sentences = [item[sentiment][i] for i in sampled_indices]
            
            samples[topic][display_name] = sampled_sentences
    
    return samples

def display_samples(samples):
    """Print samples in a readable format."""
    for topic, sentiments in samples.items():
        print(f"\n{'='*80}")
        print(f"TOPIC: {topic}")
        print(f"{'='*80}")
        
        for sentiment, sentences in sentiments.items():
            print(f"\n{sentiment.upper()}:")
            for i, sentence in enumerate(sentences, 1):
                print(f"{i}. {sentence}")
        
        print("\n")

def save_samples_to_text(samples, output_file='sampled_sentences.txt'):
    """Save samples to a text file for better readability."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for topic, stances in samples.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"TOPIC: {topic}\n")
            f.write(f"{'='*80}\n")
            
            for stance, sentences in stances.items():
                f.write(f"\n{stance.upper()} STANCE:\n")
                for i, sentence in enumerate(sentences, 1):
                    f.write(f"{i}. {sentence}\n")
            
            f.write("\n\n")
    
    print(f"Samples saved to '{output_file}'")
    return os.path.abspath(output_file)

categories = {
    "דת ומדינה": [
        "הדתה", "רדיפה דתית", "מעמד הר הבית", "תחבורה ציבורית בשבת", 
        "הפרדת דת ומדינה", "לימודי יהדות", "שמירת שבת במרחב הציבורי", 
        "חינוך ממלכתי-דתי", "כשרות במסעדות", "גיור", "נישואים אזרחיים", 
        "חופש דת", "הפרדה מגדרית", "גיוס חרדים", "שוויון בנטל", "לימודי ליבה"
    ],
    "פוליטיקה וממשל": [
        "בנימין נתניהו", "מירי רגב", "שחיתות", "בחירות", "הפגנות", 
        "רפורמה משפטית", "התערבות בג\"ץ", "עצמאות השב\"כ", "ממשלת אחדות", 
        "חוק הלאום", "תקציב המדינה", "מערכת המשפט", "שיטות הצבעה", "מדיניות אכיפה",
        "פוליטיקה", "שכר בכירים במגזר הציבורי"
    ],
    "הסכסוך הישראלי-פלסטיני": [
        "שיקום עזה", "חמאס", "התנחלויות", "הסכסוך הישראלי-פלסטיני", 
        "הסכמי שלום", "פינוי יישובים", "מעמד הר הבית", "זכויות פלסטינים", 
        "עונש מוות למחבלים", "גבולות ישראל", "ריבונות בשטחי יהודה ושומרון", 
        "מעמד ירושלים", "זכות השיבה", "שימוש בכוח בעימותים", "יוזמות שלום אזוריות", 
        "הרשות הפלסטינית", "מדינה דו-לאומית"
    ],
    "ביטחון": [
        "ביטחון", "מלחמה בטרור", "הפרות סדר של פעילי ימין", 
        "הפרות סדר של פעילי שמאל", "סוגיית השבויים", "פדיון שבויים", 
        "נשק בלתי חוקי", "ביטחון אישי", "סייבר וביטחון לאומי", 
        "קיצוץ בתקציב הביטחון", "הסדרים עם לבנון", "עיצומים על איראן", "מלחמת המפרץ"
    ],
    "כלכלה וחברה": [
        "יוקר המחיה", "משבר הדיור", "מחירי מזון", "עצמאות אנרגטית", 
        "הפרטת שירותים ציבוריים", "שכר מינימום", "אינפלציה", "מחירי דלק", 
        "פיקוח על מחירים", "צמצום פערים", "אבטלה", "צמצום בירוקרטיה", 
        "תעשיות הייטק", "מכסות יבוא", "מס עשירים", "מונופולים", "שוק ההון", 
        "העסקה קבלנית", "זכויות עובדי קבלן", "עמלות בנקאיות", "שווקים פיננסיים", 
        "אשראי צרכני", "שעות עבודה", "מדיניות דיור", "ביטוח אבטלה", 
        "פנסיה תקציבית", "דיור בר השגה", "פערי שכר מגדריים", "תיווך דירות", 
        "מדיניות מיסוי", "ענף הבנייה", "הדיור הציבורי", "עבודה מהבית", 
        "מיסוי מקרקעין", "תיירות פנים"
    ],
    "זכויות אדם ושוויון": [
        "הדרת נשים", "חופש ביטוי", "זכויות להט״ב", "עובדים זרים", 
        "מבקשי מקלט", "פער חברתי", "ייצוג נשים", "אפליה עדתית", 
        "זכויות אדם", "חופש עיסוק", "רב תרבותיות", "זכויות קשישים", 
        "זכויות ילדים", "זכויות נפגעי עבירה", "זכויות דיגיטליות", 
        "שוויון בחינוך", "ייצוג עדתי", "ועדות קבלה ביישובים", 
        "פמיניזם" 
    ],
    "תקשורת ודמוקרטיה": [
        "מניפולציות תקשורתיות", "תעמולה", "חופש העיתונות", "תקשורת", 
        "הסברה", "הסברה ישראלית", "מחאה אזרחית", "צנזורה צבאית", 
        "חינוך לדמוקרטיה", "פרטיות ברשת", "חופש אקדמי", "רשתות חברתיות", 
        "עצמאות התקשורת", "מונופול תקשורת"
    ],
    "בריאות ורווחה": [
        "כושר ותזונה", "דימוי גוף", "קורונה", "חובת חיסונים", 
        "שביתת רופאים", "ביטוח בריאות ממלכתי", "רפואה פרטית", 
        "שירותי רווחה", "רפורמה במערכת הבריאות", "ביטוח לאומי", 
        "קצבאות נכים", "הבטחת הכנסה", "הפללת צריכת קנאביס" 
    ],
    "חינוך ותרבות": [
        "ספרות", "תרבות צעירים", "שירות לאומי", "לימודי ליבה", 
        "השכלה גבוהה", "חינוך חינם מגיל אפס", "פיקוח על מסגרות חינוך", 
        "כוח אדם בחינוך", "חינוך טכנולוגי", "בתי ספר דמוקרטיים", 
        "רפורמה בחינוך", "תרבות ישראלית", "שוויון בחינוך", 
        "אלימות בבתי ספר", "שחיקת מעמד המורים", "חינוך מיני", 
        "צהרונים מסובסדים", "שביתות במערכת החינוך", "תוכניות ריאליטי",
        "מעמד העברית"
    ],
    "תחבורה ותשתיות": [
        "חניה בתל אביב", "הסעת המונים", "שיפור תשתיות", 
        "כבישי אגרה", "צמצום תאונות דרכים", "נגישות לנכים", 
        "תחבורה ותשתיות", "חניות", "תשתיות תחבורה", 
        "תחבורה ציבורית", "תחבורה פרטית", "תחבורה חכמה", 
        "כבישים", "רכבות", "אוטובוסים"
    ],
    "סביבה ואקלים": [
        "איכות הסביבה", "מתווה הגז", "התחממות גלובלית", 
        "משבר האקלים", "הגנת הסביבה", "אנרגיה מתחדשת", 
        "חקלאות ישראלית", "זיהום אוויר", "מחסור במים", 
        "ניהול משבר המים", "סובסידיות לחקלאים", "זכויות בעלי חיים"
    ],
    "מדיניות חוץ והגירה": [
        "חרם על ישראל", "מדיניות חוץ", "משבר הפליטים", 
        "קליטת עלייה", "הגירה יהודית", "מדיניות הגירה", 
        "מדיניות חוץ והגירה", "הגירה בלתי חוקית", "חוק השבות"
    ],
    "פיתוח אזורי": [
        "השקעה בפריפריה", "פיתוח הנגב והגליל", "עיר ועיירה", 
        "הכרה ביישובים בדואים", "הכרה בישובים", "פיתוח אזורי", 
        "תכנון עירוני", "תיירות פנים", "תיירות חוץ"
    ],
    "אלימות וחוק": [
        "הטרדה מינית", "אלימות במגזר הערבי", "אלימות משטרתית", 
        "זכויות חשודים", "אלימות נגד נשים", "פיקוח על בנקים",
        "אלימות וחוק", "אלימות בבתי ספר", "אלימות במשפחה", 
        "אלימות פוליטית"
    ],
    "חדשנות וטכנולוגיה": [
        "אבטחת מידע", "תרבות הסטארט-אפ", "חדשנות וטכנולוגיה", 
        "יזמות", "הייטק", "תעשייה מסורתית", "תעשיות עתירות ידע", 
        "תעשיות כבדות", "תעשיות קלות", "תעשיות טכנולוגיות",
    ],
    "ספורט": [
        "הישגיות בספורט", "ספורט תחרותי", "ספורט קבוצתי", 
        "ספורט אישי", "אולימפיאדה", "ספורט מקצועי", 
        "ספורט חובבני", "אימון ספורט", "תזונה בספורט", "בריאות בספורט"
    ]
}


topic_to_category = {}
for category, topics in categories.items():
    for topic in topics:
        topic_to_category[topic] = category

def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"שגיאה בטעינת קובץ JSON: {e}")
        return None

def categorize_topics(data):
    results = {category: [] for category in categories}
    uncategorized = []
    
    for item in data:
        topic = item['topic']
        category = topic_to_category.get(topic)
        
        if category:
            results[category].append({
                'topic': topic,
                'supportive_sentences': item['supportive_sentences'],
                'opposing_sentences': item['opposing_sentences'],
                'neutral_sentences': item['neutral_sentences']
            })
        else:
            uncategorized.append(topic)
    
    return results, uncategorized

def save_categorized_data(categorized_data, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(categorized_data, file, ensure_ascii=False, indent=4)
        print(f"הנתונים נשמרו בהצלחה ב-{output_file}")
    except Exception as e:
        print(f"שגיאה בשמירת הנתונים: {e}")


def main():
    # File path to your data
    file_path = './Data/batch_output.json'  
    # output_text_file='./Data/sampled_sentences.txt'
    output_file = './Data/categorized_topics_.json'

    data = load_json_data(file_path)
    if not data:
        return
    
    categorized_data, uncategorized = categorize_topics(data)
    
    if uncategorized:
        print("נושאים שלא קוטלגו:")
        for topic in uncategorized:
            print(f"- {topic}")
    
    category_counts = {category: len(topics) for category, topics in categorized_data.items()}
    print("\nמספר הנושאים בכל קטגוריה:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")
    
    save_categorized_data(categorized_data, output_file)
    
    # try:
    #     # Load data from the provided content
    #     data = load_data(file_path)
        
    #     # Sample sentences
    #     samples = sample_sentences(data)
        
    #     # Display samples
    #     # display_samples(samples)

    #     # Save samples to text file
    #     output_file_path = save_samples_to_text(samples, output_text_file)
    #     print(f"Full samples saved to: {output_file_path}")
        
    #     # Optional: Save samples to CSV for further analysis
    #     save_to_csv = False  # Change to True if needed
    #     if save_to_csv:
    #         rows = []
    #         for topic, stances in samples.items():
    #             for stance, sentences in stances.items():
    #                 for sentence in sentences:
    #                     rows.append({
    #                         'topic': topic,
    #                         'stance': stance,
    #                         'sentence': sentence
    #                     })
            
    #         df = pd.DataFrame(rows)
    #         df.to_csv('sampled_sentences.csv', index=False, encoding='utf-8-sig')
    #         print("Samples saved to 'sampled_sentences.csv'")
    
    # except Exception as e:
    #     print(f"Error: {e}")

if __name__ == "__main__":
    # main() # 1

    # # create csv file 
    # # 2
    # path = './Data/categorized_topics_.json'
    # try:
    #     with open(path, 'r', encoding='utf-8') as file:
    #         data = json.load(file)
    # except Exception as e:
    #     print(f"Error in loading JSON file: {e}")

    # # Prepare list to store rows for CSV
    # csv_data = []

    # # Iterate over the main topics
    # for main_topic, sub_topics in data.items():
    #     for sub_topic in sub_topics:
    #         topic = sub_topic["topic"]
            
    #         # Collect sentences for each stance type (supportive, opposing, neutral)
    #         for sentence in sub_topic["supportive_sentences"]:
    #             csv_data.append([main_topic, topic, sentence, "support"])
    #         for sentence in sub_topic["opposing_sentences"]:
    #             csv_data.append([main_topic, topic, sentence, "against"])
    #         for sentence in sub_topic["neutral_sentences"]:
    #             csv_data.append([main_topic, topic, sentence, "neutral"])

    # # Convert to a DataFrame
    # df = pd.DataFrame(csv_data, columns=["Topic", "Sub-topic", "Text", "Stance"])

    # # Save to CSV
    # df.to_csv("./Data/Hebrew_stance_dataset_.csv", index=False, encoding="utf-8")

    # print("CSV file has been created successfully.")


    # # 3
    # # Read the original CSV
    # df = pd.read_csv('./Data/Hebrew_stance_dataset_.csv')
    # print("Loaded CSV file successfully.")

    # # Map 'support' to 'תומך' in the Stance column
    # stance_mapping = {'support': 'תומך', 'against': 'מתנגד', 'neutral': 'נייטרלי'}
    # df['Stance'] = df['Stance'].map(stance_mapping)

    # # Create a new Topic field by appending the stance in Hebrew
    # df['Topic'] = df['Topic'] + '_' + df['Stance']

    # # Save the transformed data
    # df.to_csv('./Data/Hebrew_stance_dataset_modified_.csv', index=False)
    # print("Transformed CSV file has been created successfully.")

    # 4
    # concat all csv files in the directory
    # df1 = pd.read_csv('./Data/Hebrew_stance_dataset_modified_.csv')
    # df2 = pd.read_csv('./Data/Hebrew_stance_dataset_modified.csv')

    # print("Loaded CSV files successfully.")

    # combined_df = pd.concat([df1, df2], ignore_index=True)

    # combined_df.to_csv('./Data/Hebrew_stance_dataset_combined.csv', index=False)

    # print(f"Combined dataset contains {len(combined_df)} rows")




    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # new_data = pd.read_csv('./Data/datasets/HF_Gen_data.csv')
    # print("Loaded CSV file successfully.")
    # # new_data['Topic'] = new_data['Topic'].replace('תקשורת_نייטרלי', 'תקשורת_נייטרלי')
    # # new_data['Topic'] = new_data['topic'] + '_' + new_data['stance']
    # # new_data = new_data.rename(columns={'sentence': 'Text'})
    

    # print(new_data.head())
    # new_data.to_csv('./Data/datasets/HF_Gen_data.csv', index=False)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # df = pd.read_csv('./Data/datasets/Hebrew_stance_dataset_modified.csv')
    # print("Loaded CSV file successfully.")

    # # Map 'support' to 'תומך' in the Stance column
    # stance_mapping = {'תומך': 'בעד', 'מתנגד': 'נגד', 'נייטרלי': 'נייטרלי'}
    # df['Stance'] = df['Stance'].map(stance_mapping)

    # new_df = pd.DataFrame(columns=['sentence','topic','stance'])
    # new_df['sentence'] = df['Text']
    # new_df['topic'] = df['Sub-topic']
    # new_df['stance'] = df['Stance']

    # # Save the transformed data
    # new_df.to_csv('./Data/topic_stance_dataset_synthetic.csv', index=False)
    # print("Transformed CSV file has been created successfully.")

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # df1 = pd.read_csv('./Data/topic_stance_dataset_synthetic.csv')
    # df2 = pd.read_csv('./Data/topic_stance_dataset.csv')
    # print("Loaded CSV files successfully.")
    # combined_df = pd.concat([df1, df2], ignore_index=True)
    # combined_df = combined_df.drop_duplicates(subset=['sentence', 'topic', 'stance'], keep='first')
    # combined_df.to_csv('./Data/topic_stance_dataset_combined.csv', index=False)
    # print(f"Combined dataset contains {len(combined_df)} rows")

    df_comb = pd.read_csv('./Data/datasets/topic_stance_dataset_combined.csv')
    # shuffle the dataframe
    df_comb = df_comb.sample(frac=1).reset_index(drop=True)
    # save the shuffled dataframe
    df_comb.to_csv('./Data/datasets/topic_stance_dataset_combined_shuffled.csv', index=False)
    print(f"Shuffled dataset contains {len(df_comb)} rows")