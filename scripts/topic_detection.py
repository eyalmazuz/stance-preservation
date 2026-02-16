from src.data_loader import load_data
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re
import json
from src.utils import split_into_sentences

def get_topic_for_model(context, hebrew_sentence, model, tokenizer, topic_detection_model):
    """Get topic for a Hebrew sentence using the model."""
    dynamic_examples = [
        {
            "משפט": "עולה מניתוח של 7,500 מחקרים שפורסמו בנושא בין השנים 1973 עד 2011 כי ב־40 השנים האחרונות יש מגמת ירידה מתמשכת בפוריות הגבר בעולם המערבי ונרשמה ירידה של יותר מ־50 אחוזים בריכוז ובספירת הזרע",
            "ניתוח": "מדובר על פוריות הגבר בעולם המערבי, וירידה בריכוז ובספירת הזרע.",
            "נושא": "פוריות הגבר"
        },
        {
            "משפט": "עם עומסי החום של הקיץ והגידול בשימוש במזגנים, עולות גם התקלות, ומדריך זה סוקר את התקלות הנפוצות, עלויות התיקון והמלצות לבחירת מזגן",
            "ניתוח": "מדובר על מזגנים, תקלות נפוצות ועלויות תיקון.",
            "נושא": "תחזוקת מזגנים "
        },
        {
            "משפט": "משרד הבריאות פרסם את נתוני התחלואה בקורונה: אחוז החיוביים ירד; בישראל יש 8,310 חולים פעילים; סך המחלימים עומד על 325,862; מספר הנפטרים מפרוץ המגפה עומד על 2,735",
            "ניתוח": "מדובר על נתוני תחלואה בקורונה בישראל",
            "נושא": "קורונה"
        },
        {
            "משפט": "כחלק מהפתרונות היצירתיים הללו חוזרים אלינו לאחרונה משחקי הילדות של דור ההורים שלא ידע אינטרנט, טאבלט, אייפון ואקס בוקס, והסתפק במשחקי רחוב עם שאר הילדים",
            "ניתוח": "מדובר על משחקי הילדות של דור ההורים, משחקים ללא טכנולוגיה מודרנית.",
            "נושא": "משחקים"
        },
        {
            "משפט": "ישראל הצהירה כי לא תאפשר שיקום עזה ללא פתרון לסוגיית השבויים והנעדרים, אך נותר לראות אם תצליח לעמוד בהבטחתה",
            "ניתוח": "מדובר על הצהרת ישראל בנוגע לשיקום עזה ולסוגיית השבויים והנעדרים.",
            "נושא": "שיקום רצועת עזה"
        },
        {
            "משפט": "מאז שהמחיר ביניהם הושווה, רכבי הפנאי מזנבים במכירות המשפחתיים, כשיבואני המשפחתיים נאלצים להוריד מחירים או להעלות ברמת האבזור",
            "ניתוח": "מדובר על רכבי פנאי לעומת רכבים משפחתיים, והשפעת המחיר על המכירות.",
            "נושא": "רכבי פנאי"
        },
        {
            "משפט": "הפטרייה, קורדיספס שמה, משתלטת על מוחן של נמלים וכופה עליהן לטפס לגובה רב כדי לפזר את נבגיה",
            "ניתוח": "מדובר על פטרייה בשם קורדיספס שמשפיעה על נמלים.",
            "נושא": "קורדיספס"
        }
    ]

    base_prompt = """הוראות:
        בהינתן טקסט, ומשפט ממנו,
        עליך לקרוא את המשפט, לנתח אותו בקצרה, ולאחר מכן להחזיר את הנושא המרכזי שבו המשפט עוסק - השתמש בטקסט כקונטקסט.

        הגדרות:
        הנושא הוא התחום המרכזי של המשפט (למשל: פוליטיקה, רפואה, חינוך, ספורט, כלכלה, ביטחון, טכנולוגיה ועוד).
        אל תיתן יותר מנושא אחד.
        הנושא צריך להיות מילה אחת או ביטוי קצר (עד 3 מילים).
        אין צורך בניסוחים כמו "הנושא הוא" - כתוב רק את הנושא.
        אם לא ניתן לזהות נושא - כתוב: לא ברור.
        כאשר קיימת ישות פועלת (למשל: "ישראל הודיעה כי..."), זהה את התחום שבו עוסקת ההצהרה, ולא את שם הגוף הפועל.
        יש מקרים בהם תצטרך להיות כללי יותר או ספציפי יותר, בהתאם למשמעות המשפט.
        אם המשפט עוסק בכמה נושאים, בחר את הנושא המרכזי ביותר.
        אם הנושא שמצאת הוא ברבים - הפוך אותו ליחיד.

        שלבי עבודה:
        1. נתח את משמעות המשפט.
        2. זהה על איזה תחום עוסק המשפט.
        3. החזר את הנושא.
        """

    examples_prompt = "\n".join([
        f'משפט: {ex["משפט"]}\nניתוח: {ex["ניתוח"]}\nנושא: {ex["נושא"]}'
        for ex in dynamic_examples
    ])

    final_prompt = (
        base_prompt
        + "\n\nקונטקסט:" + context
        + "\n\nדוגמאות:\n" + examples_prompt
        + f"\n\nמשפט: {hebrew_sentence}\nניתוח:\nנושא:"
    ).strip()

    # finetuned dont need few shots
    finetuned_prompt = (
        base_prompt
        + "\n\nקונטקסט:" + context
        + f"\n\nמשפט: {hebrew_sentence}\nניתוח:\nנושא:"
    ).strip()
    
    if topic_detection_model == 'finetuned':
        # Encode PROMPT ONLY
        inputs = tokenizer(finetuned_prompt, return_tensors="pt").to(model.device)
        prompt_length = inputs.input_ids.shape[1]

        # Generate continuation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

        # Extract only the generated tokens
        new_tokens = outputs[0][prompt_length:]
        topic = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Only return first line (the topic)
        topic = topic.split("\n")[0].strip()

    elif topic_detection_model == 'dicta-il/dictalm2.0':
        inputs = tokenizer(final_prompt.strip(), return_tensors='pt', padding=True).to(model.device)
        
        # Get the length of the prompt in tokens
        prompt_length = inputs.input_ids.shape[1]
        
        # Generate only the new tokens (the topic)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                do_sample=False,
                max_new_tokens=10,
                # temperature=0.1
            )
        
        # Extract only the new tokens (excluding the prompt)
        topic_tokens = outputs[0][prompt_length:]
        
        # Decode only the topic tokens
        topic = tokenizer.decode(topic_tokens, skip_special_tokens=True).strip()
        topic = topic.split("\n")[0]
        # topic = topic.split("\n")[0].replace("נושא:", "").strip().rstrip(".").rstrip(":")
        
    
    else:
        final_prompt = (
            base_prompt
            + "\n\nקונטקסט:" + context
            + "\n\nדוגמאות:\n" + examples_prompt
            + f"\n\nמשפט: {hebrew_sentence}\nניתוח:\nנושא:"
        )

        # Build chat messages
        messages = [{"role": "user", "content": final_prompt}]

        # IMPORTANT: tokenize=True
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=True   
        ).to(model.device)

        prompt_length = inputs.shape[1]

        outputs = model.generate(
            inputs,
            max_new_tokens=20,
            do_sample=False
        )

        # Slice away the prompt
        new_tokens = outputs[0][prompt_length:]
        topic = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # keep only the first line
        topic = topic.split("\n")[0].strip()

    return {"sentence": hebrew_sentence, "topic": topic}

def get_topic_for_model_eng(context, english_sentence, model, tokenizer, topic_detection_model):
    """Get topic for a English sentence using the model."""
    dynamic_examples = [
        {
            "sentence": "It emerges from an analysis of 7,500 studies published on the subject between the years 1973 and 2011 that over the past 40 years there has been an ongoing downward trend in male fertility in the Western world, and a decline of more than 50 percent has been recorded in sperm concentration and sperm count.",
            "analysis": "This refers to male fertility in the Western world, and a decline in sperm concentration and sperm count.",
            "topic": "Male fertility"
        },
        {
            "sentence": "With the heat loads of summer and the increase in the use of air conditioners, malfunctions also increase, and this guide reviews the common malfunctions, repair costs, and recommendations for choosing an air conditioner.",
            "analysis": "This concerns air conditioners, common malfunctions, and repair costs.",
            "topic": "Air conditioning maintenance"
        },
        {
            "sentence": "The Ministry of Health published the coronavirus morbidity data: the percentage of positive cases has decreased; in Israel there are 8,310 active patients; the total number of recovered stands at 325,862; the number of deaths since the outbreak of the pandemic stands at 2,735.",
            "analysis": "This concerns coronavirus morbidity data in Israel.",
            "topic": "Covid-19"
        },
        {
            "sentence": "As part of these creative solutions, the childhood games of the parents’ generation—who did not know the internet, tablets, iPhones, or Xbox, and made do with street games with other children—have recently been returning to us.",
            "analysis": "This refers to the childhood games of the parents’ generation, games without modern technology.",
            "topic": "Games"
        },
        {
            "sentence": "Israel declared that it will not allow the rehabilitation of Gaza without a solution to the issue of the captives and the missing, but it remains to be seen whether it will succeed in upholding its promise.",
            "analysis": "This refers to Israel’s declaration regarding the rehabilitation of Gaza and the issue of the captives and the missing.",
            "topic": "Gaza rehabilitation"
        },
        {
            "sentence": "Since the price between them was equalized, leisure vehicles have been nipping at the heels of family cars in sales, with importers of family cars being forced to lower prices or raise the level of equipment.",
            "analysis": "This concerns leisure vehicles versus family cars, and the effect of price on sales.",
            "topic": "SUVs"
        },
        {
            "sentence": "The fungus, cordyceps by name, takes over the brains of ants and compels them to climb to great heights in order to disperse its spores.",
            "analysis": "This refers to a fungus called cordyceps that affects ants.",
            "topic": "Cordyceps"
        }
    ]

    base_prompt = """
    Instructions:
    Given a context and a sentence from it,
    you need to read the sentence, analyze it briefly, and then return the main topic the sentence deals with - use the context as reference.

    Definitions:
    The topic is the main field of the sentence (e.g., politics, medicine, education, sports, economy, security, technology, etc.).
    Do not provide more than one topic.
    The topic should be a single word or a short phrase (up to 3 words).
    There is no need for phrases like "the topic is" - just write the topic.
    If the topic cannot be identified - write: unclear.
    When there is an acting entity (e.g., "Israel announced that..."), identify the field the statement deals with, not always the name of the acting body.
    There are cases where you need to be more general or more specific, depending on the meaning of the sentence.
    If the sentence deals with several topics, choose the most central topic.
    If the topic you found is plural - convert it to singular.

    Work Steps:
    1. Analyze the meaning of the sentence.
    2. Identify which field the sentence deals with.
    3. Return the topic.
    """

    examples_prompt = "\n".join([
        f'Sentence: {ex["sentence"]}\nAnalysis: {ex["analysis"]}\nTopic: {ex["topic"]}'
        for ex in dynamic_examples
    ])

    final_prompt = (
        base_prompt
        + "\n\nContext:" + context
        + "\n\nExamples:\n" + examples_prompt
        + f"\n\nSentence: {english_sentence}\nAnalysis:\nTopic:"
    ).strip()

    # finetuned dont need few shots
    finetuned_prompt = (
        base_prompt
        + "\n\nContext:" + context
        + f"\n\nSentence: {english_sentence}\nAnalysis:\nTopic:"
    ).strip()
    
    if topic_detection_model == 'finetuned':
        # Encode PROMPT ONLY
        inputs = tokenizer(finetuned_prompt, return_tensors="pt").to(model.device)
        prompt_length = inputs.input_ids.shape[1]

        # Generate continuation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

        # Extract only the generated tokens
        new_tokens = outputs[0][prompt_length:]
        topic = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Only return first line (the topic)
        topic = topic.split("\n")[0].strip()

    elif topic_detection_model == 'dicta-il/dictalm2.0':
        inputs = tokenizer(final_prompt.strip(), return_tensors='pt', padding=True).to(model.device)
        
        # Get the length of the prompt in tokens
        prompt_length = inputs.input_ids.shape[1]
        
        # Generate only the new tokens (the topic)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                do_sample=False,
                max_new_tokens=30,
                # temperature=0.1
            )
        
        # Extract only the new tokens (excluding the prompt)
        topic_tokens = outputs[0][prompt_length:]
        
        # Decode only the topic tokens
        topic = tokenizer.decode(topic_tokens, skip_special_tokens=True).strip()
        topic = topic.split("\n")[0]
        # topic = topic.split("\n")[0].replace("נושא:", "").strip().rstrip(".").rstrip(":")
        
    
    else:
        final_prompt = (
            base_prompt
            + "\n\nContext:" + context
            + "\n\nExamples:\n" + examples_prompt
            + f"\n\nSentence: {english_sentence}\nAnalysis:\nTopic:"
        )

        messages = [
            {"role": "user", "content": final_prompt}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        prompt_length = inputs.shape[1]

        outputs = model.generate(
            inputs,
            max_new_tokens=30,
            do_sample=False,
        )

        # print("Prompt length:", prompt_length)
        # print("Output length:", outputs[0].shape[0])


        new_tokens = outputs[0][prompt_length:]
        topic = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().split("\n")[0].strip()

        # keep only the first line
        topic = topic.split("\n")[0].strip()

    return {"sentence": english_sentence, "topic": topic}
