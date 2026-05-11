def get_hebrew_prompt_template() -> str:
    dynamic_examples = [
        {
            "משפט": (
                "עולה מניתוח של 7,500 מחקרים שפורסמו בנושא בין השנים 1973 עד 2011 כי "
                "ב־40 השנים האחרונות יש מגמת ירידה מתמשכת בפוריות הגבר בעולם המערבי "
                "ונרשמה ירידה של יותר מ־50 אחוזים בריכוז ובספירת הזרע"
            ),
            # "ניתוח": "מדובר על פוריות הגבר בעולם המערבי, וירידה בריכוז ובספירת הזרע.",
            "נושא": "פוריות הגבר",
        },
        {
            "משפט": (
                "עם עומסי החום של הקיץ והגידול בשימוש במזגנים, עולות גם התקלות, "
                "ומדריך זה סוקר את התקלות הנפוצות, עלויות התיקון והמלצות לבחירת מזגן"
            ),
            # "ניתוח": "מדובר על מזגנים, תקלות נפוצות ועלויות תיקון.",
            "נושא": "תחזוקת מזגנים ",
        },
        {
            "משפט": (
                "משרד הבריאות פרסם את נתוני התחלואה בקורונה: אחוז החיוביים ירד; "
                "בישראל יש 8,310 חולים פעילים; סך המחלימים עומד על 325,862; "
                "מספר הנפטרים מפרוץ המגפה עומד על 2,735"
            ),
            # "ניתוח": "מדובר על נתוני תחלואה בקורונה בישראל",
            "נושא": "קורונה",
        },
        {
            "משפט": (
                "כחלק מהפתרונות היצירתיים הללו חוזרים אלינו לאחרונה משחקי הילדות "
                "של דור ההורים שלא ידע אינטרנט, טאבלט, אייפון ואקס בוקס, "
                "והסתפק במשחקי רחוב עם שאר הילדים"
            ),
            # "ניתוח": "מדובר על משחקי הילדות של דור ההורים, משחקים ללא טכנולוגיה מודרנית.",
            "נושא": "משחקים",
        },
        {
            "משפט": (
                "ישראל הצהירה כי לא תאפשר שיקום עזה ללא פתרון לסוגיית השבויים "
                "והנעדרים, אך נותר לראות אם תצליח לעמוד בהבטחתה"
            ),
            # "ניתוח": "מדובר על הצהרת ישראל בנוגע לשיקום עזה ולסוגיית השבויים והנעדרים.",
            "נושא": "שיקום רצועת עזה",
        },
        {
            "משפט": (
                "מאז שהמחיר ביניהם הושווה, רכבי הפנאי מזנבים במכירות המשפחתיים, "
                "כשיבואני המשפחתיים נאלצים להוריד מחירים או להעלות ברמת האבזור"
            ),
            # "ניתוח": "מדובר על רכבי פנאי לעומת רכבים משפחתיים, והשפעת המחיר על המכירות.",
            "נושא": "רכבי פנאי",
        },
        {
            "משפט": "הפטרייה, קורדיספס שמה, משתלטת על מוחן של נמלים וכופה עליהן לטפס לגובה רב כדי לפזר את נבגיה",
            # "ניתוח": "מדובר על פטרייה בשם קורדיספס שמשפיעה על נמלים.",
            "נושא": "קורדיספס",
        },
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

    examples_prompt = "\n".join([f"משפט: {ex['משפט']}\nנושא: {ex['נושא']}" for ex in dynamic_examples])

    final_prompt = (
        base_prompt + "\n\nקונטקסט:" + "{context}" + "\n\nדוגמאות:\n" + examples_prompt + "\n\nמשפט: {sentence}\nנושא:"
    ).strip()

    return final_prompt


def get_hebrew_prompt_template_original_examples() -> str:
    dynamic_examples = [
        # {
        #     "משפט": (
        #         "העירייה הודיעה כי החל מהחודש הבא ייפתחו שלושה מרכזי תיקון "
        #         "לאופניים חשמליים כדי לצמצם פסולת אלקטרונית ולעודד שימוש חוזר."
        #     ),
        #     "נושא": "מחזור אופניים",
        # },
        # {
        #     "משפט": (
        #         "חוקרים מצאו כי חשיפה ממושכת לאור כחול בשעות הערב פוגעת באיכות "
        #         "השינה של מתבגרים ומגבירה עייפות בבוקר."
        #     ),
        #     "נושא": "שינה מתבגרים",
        # },
        # {
        #     "משפט": "רשות הטבע סגרה זמנית את מסלול ההליכה בנחל לאחר שנמדדה בו עלייה חריגה בזיהום ממי ביוב.",
        #     "נושא": "זיהום נחלים",
        # },
        # {
        #     "משפט": "רשת המרכולים השיקה עגלות חכמות שמזהות מוצרים אוטומטית ומאפשרות ללקוחות לצאת ללא מעבר בקופה.",
        #     "נושא": "עגלות חכמות",
        # },
        # {
        #     "משפט": "בית החולים החל להפעיל מערכת רובוטית שמסייעת לרוקחים להכין מנות כימותרפיה בדיוק גבוה יותר.",
        #     "נושא": "רוקחות רובוטית",
        # },
        # {
        #     "משפט": "חברת התעופה תציע החל בקיץ חבילות מוזלות לנוסעים שמוותרים על מזוודה ובוחרים בטיסות לילה.",
        #     "נושא": "תמחור טיסות",
        # },
        {
            "משפט": (
                "משרד החינוך ירחיב את תוכנית הגינות הלימודיות כך שכל תלמידי כיתה ד' יגדלו ירקות כחלק משיעורי המדעים."
            ),
            "נושא": "גינות לימודיות",
        },
        {
            "משפט": "בבדיקה שנערכה בנמל נמצא כי עיכובים בפריקת מכולות נגרמים בעיקר ממחסור בנהגי משאיות בשעות הלילה.",
            "נושא": "פריקת מכולות",
        },
        {
            "משפט": "מפתחים ישראלים הציגו שבב חדש שמאפשר להפעיל מודלי בינה מלאכותית על רחפנים ללא חיבור לענן.",
            "נושא": "שבבי רחפנים",
        },
        {
            "משפט": "הספרייה העירונית פתחה מסלול השאלה מהיר שמאפשר להזמין ספרים באפליקציה ולאסוף אותם מתא חכם ברחוב.",
            "נושא": "השאלת ספרים",
        },
        {
            "משפט": "התאחדות החקלאים מזהירה כי מחסור בעובדי קטיף צפוי להוביל לעלייה במחירי האפרסקים כבר בתחילת העונה.",
            "נושא": "עובדי קטיף",
        },
        {
            "משפט": (
                "מיזם חדש בבאר שבע מחבר בין סטודנטים לקשישים לצורך ביקורי בית שבועיים והדרכה בשימוש בשירותים דיגיטליים."
            ),
            "נושא": "סיוע דיגיטלי",
        },
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

    examples_prompt = "\n".join([f"משפט: {ex['משפט']}\nנושא: {ex['נושא']}" for ex in dynamic_examples])

    final_prompt = (
        base_prompt + "\n\nקונטקסט:" + "{context}" + "\n\nדוגמאות:\n" + examples_prompt + "\n\nמשפט: {sentence}\nנושא:"
    ).strip()

    return final_prompt


def get_english_prompt_template() -> str:
    dynamic_examples = [
        {
            "sentence": (
                "It emerges from an analysis of 7,500 studies published on the subject between the years 1973 "
                "and 2011 that over the past 40 years there has been an ongoing downward trend in male fertility "
                "in the Western world, and a decline of more than 50 percent has been recorded in sperm "
                "concentration and sperm count."
            ),
            "analysis": (
                "This refers to male fertility in the Western world, and a decline in sperm concentration and sperm "
                "count."
            ),
            "topic": "Male fertility",
        },
        {
            "sentence": (
                "With the heat loads of summer and the increase in the use of air conditioners, malfunctions also "
                "increase, and this guide reviews the common malfunctions, repair costs, and recommendations for "
                "choosing an air conditioner."
            ),
            "analysis": "This concerns air conditioners, common malfunctions, and repair costs.",
            "topic": "Air conditioning maintenance",
        },
        {
            "sentence": (
                "The Ministry of Health published the coronavirus morbidity data: the percentage of positive cases "
                "has decreased; in Israel there are 8,310 active patients; the total number of recovered stands at "
                "325,862; the number of deaths since the outbreak of the pandemic stands at 2,735."
            ),
            "analysis": "This concerns coronavirus morbidity data in Israel.",
            "topic": "Covid-19",
        },
        {
            "sentence": (
                "As part of these creative solutions, the childhood games of the parents' generation, who did not "
                "know the internet, tablets, iPhones, or Xbox, and made do with street games with other children, "
                "have recently been returning to us."
            ),
            "analysis": (
                "This refers to the childhood games of the parents' generation, games without modern technology."
            ),
            "topic": "Games",
        },
        {
            "sentence": (
                "Israel declared that it will not allow the rehabilitation of Gaza without a solution to the issue "
                "of the captives and the missing, but it remains to be seen whether it will succeed in upholding "
                "its promise."
            ),
            "analysis": (
                "This refers to Israel's declaration regarding the rehabilitation of Gaza and the issue of the "
                "captives and the missing."
            ),
            "topic": "Gaza rehabilitation",
        },
        {
            "sentence": (
                "Since the price between them was equalized, leisure vehicles have been nipping at the heels of "
                "family cars in sales, with importers of family cars being forced to lower prices or raise the "
                "level of equipment."
            ),
            "analysis": "This concerns leisure vehicles versus family cars, and the effect of price on sales.",
            "topic": "SUVs",
        },
        {
            "sentence": (
                "The fungus, cordyceps by name, takes over the brains of ants and compels them to climb to great "
                "heights in order to disperse its spores."
            ),
            "analysis": "This refers to a fungus called cordyceps that affects ants.",
            "topic": "Cordyceps",
        },
    ]

    base_prompt = """
    Instructions:
    Given a context and a sentence from it,
    you need to read the sentence, analyze it briefly, and then return the main topic the sentence deals with
    - use the context as reference.

    Definitions:
    The topic is the main field of the sentence
    (e.g., politics, medicine, education, sports, economy, security, technology, etc.).
    Do not provide more than one topic.
    The topic should be a single word or a short phrase (up to 3 words).
    There is no need for phrases like "the topic is" - just write the topic.
    If the topic cannot be identified - write: unclear.
    When there is an acting entity (e.g., "Israel announced that..."), identify the field the statement deals with,
    not always the name of the acting body.
    There are cases where you need to be more general or more specific, depending on the meaning of the sentence.
    If the sentence deals with several topics, choose the most central topic.
    If the topic you found is plural - convert it to singular.

    Work Steps:
    1. Analyze the meaning of the sentence.
    2. Identify which field the sentence deals with.
    3. Return the topic.
    """

    examples_prompt = "\n".join(
        f"Sentence: {example['sentence']}\nAnalysis: {example['analysis']}\nTopic: {example['topic']}"
        for example in dynamic_examples
    )

    final_prompt = (
        base_prompt
        + "\n\nContext:"
        + "{context}"
        + "\n\nExamples:\n"
        + examples_prompt
        + "\n\nSentence: {sentence}\nAnalysis:\nTopic:"
    ).strip()

    return final_prompt


def get_prompt(language: str) -> str:
    if language == "he":
        return get_hebrew_prompt_template()
    elif language == "en":
        return get_english_prompt_template()
    else:
        raise ValueError(f"Invalid langauge: {language}")


def get_emd_prompt(language: str) -> str:
    if language == "he":
        return get_hebrew_prompt_template_original_examples()
    elif language == "en":
        return get_english_prompt_template()
    else:
        raise ValueError(f"Invalid langauge: {language}")
