"""
Hebrew stance classification (בעד/נגד/נייטרלי) toward a topic using zero-shot NLI (XLM-R XNLI).
Computes stance for article & summary and their shift (summary - article).
"""
import pandas as pd
from typing import Dict, List, Tuple
from transformers import pipeline
from scipy.stats import pearsonr, spearmanr, kendalltau


# data_path = "./tests/output/labeled_data_with_scores.csv"
data_path = "./tests/output/english_labeled_data_with_scores.csv"
stance_model_name = "joeddav/xlm-roberta-large-xnli"

# Columns in your CSV
sum_col = "Summary"
art_col = "Article"
# sum_col = "Sentence in Summary"
# art_col = "Best Match Sentences From Article"
art_topic_col = "article topic"
sum_topic_col = "summary topic"

# ======= Output label set (Hebrew) =======
STANCE_LABELS_HE = ["בעד", "נגד", "נייטרלי"]
# STANCE_TO_SCORE_HE = {"בעד": +1.0, "נגד": -1.0, "נייטרלי": 0.0}
STANCE_TO_SCORE_HE = {"Favor": +1.0, "Against": -1.0, "Neutral": 0.0}

# ======= Optional: English candidate labels (more robust sometimes) =======
USE_EN_CANDIDATES = True # False
STANCE_LABELS_EN = ["Favor", "Against", "Neutral"]
# mapping from EN prediction -> Hebrew for saving
EN_TO_HE = {
    "Favor": "בעד",
    "Against": "נגד",
    "Neutral": "נייטרלי",
}

def build_stance_classifier():
    """
    Returns a zero-shot classifier pipeline for stance.
    """
    # If you want GPU auto-placement, add device_map="auto" (requires accelerate)
    return pipeline(
        "zero-shot-classification",
        model=stance_model_name,
        # tokenizer/model truncation is on by default; this kwarg is a harmless hint
        truncation=True,
    )

def score_distribution_to_expected(labels: List[str], scores: List[float], value_map: Dict[str, float]) -> float:
    """
    Expected-value stance in [-1, 1] from the zero-shot distribution.
    labels: returned candidate labels (same strings as passed in candidate_labels)
    value_map: maps those labels to numeric scores
    """
    total = float(sum(scores)) or 1.0
    s = 0.0
    for lab, sc in zip(labels, scores):
        s += value_map.get(lab, 0.0) * float(sc)
    # clip for safety
    v = s / total
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)

def classify_stance(clf, text: str, topic: str = "") -> Tuple[str, float, Dict[str, float]]:
    """
    Returns (discrete_label_hebrew, expected_score[-1..1], per_label_probs_hebrew)
    where discrete_label_hebrew ∈ {"בעד","נגד","נייטרלי"}.
    English where discrete_label_hebrew ∈ {"Favor","Against","Neutral"}.
    """
    if not isinstance(text, str) or not text.strip():
        # return ("נייטרלי", 0.0, {"בעד": 0.0, "נגד": 0.0, "נייטרלי": 1.0})
        return ("Neutral", 0.0, {"Favor": 0.0, "Against": 0.0, "Neutral": 1.0})
    

    topic_str = topic.strip() if isinstance(topic, str) else ""
    # Hypothesis templates — Hebrew by default.
    if USE_EN_CANDIDATES:
        # English candidate labels; Hebrew hypothesis still works, but English hypothesis can be even more stable.
        hyp = f"This text expresses a {{}} stance toward '{topic_str or 'the topic'}'."
        candidate_labels = STANCE_LABELS_EN
    else:
        hyp = f"הטקסט הבא מביע עמדה {{}} כלפי '{topic_str or 'הנושא'}'."
        candidate_labels = STANCE_LABELS_HE

    res = clf(
        text,
        candidate_labels=candidate_labels,
        hypothesis_template=hyp,
        multi_label=False,
    )
    # res["labels"] sorted by descending score; res["scores"] aligned.
    labels = list(res["labels"])
    scores = [float(x) for x in res["scores"]]

    # if USE_EN_CANDIDATES:
    #     # Map EN -> HE for output and scoring
    #     labels_he = [EN_TO_HE[l] for l in labels]
    #     value_map = {EN_TO_HE[k]: v for k, v in zip(STANCE_LABELS_EN, [+1.0, -1.0, 0.0])}
    #     top_label_he = EN_TO_HE[labels[0]]
    #     exp_score = score_distribution_to_expected(labels_he, scores, value_map)
    #     probs_he = {lab_he: float(sc) for lab_he, sc in zip(labels_he, scores)}
    #     return (top_label_he, exp_score, probs_he)
    # else:
    value_map = {k: v for k, v in STANCE_TO_SCORE_HE.items()}
    top_label_he = labels[0]
    exp_score = score_distribution_to_expected(labels, scores, value_map)
    probs_he = {lab: float(sc) for lab, sc in zip(labels, scores)}
    return (top_label_he, exp_score, probs_he)
    

def corr_table_with_pvals(data, cols, method="pearson"):
    out = pd.DataFrame(index=cols, columns=cols, dtype=object)
    for i in cols:
        for j in cols:
            if method == "pearson":
                r, p = pearsonr(data[i], data[j])
            elif method == "spearman":
                r, p = spearmanr(data[i], data[j])
            elif method == "kendall":
                r, p = kendalltau(data[i], data[j])
            out.loc[i, j] = f"{r:.3f} (p={p:.3g})"
    return out

if __name__ == "__main__":
    # Load data
    df = pd.read_csv(data_path)

    # Handle NaNs
    for c in [sum_col, art_col, art_topic_col, sum_topic_col]:
        if c in df.columns:
            df[c] = df[c].fillna("")

    # Build stance classifier
    stance_clf = build_stance_classifier()

    # Article stance
    if art_topic_col in df.columns:
        art_results = df.apply(
            lambda r: classify_stance(stance_clf, r[art_col], r[art_topic_col]),
            axis=1,
        )
    else:
        art_results = df[art_col].apply(lambda x: classify_stance(stance_clf, x, ""))

    df["article_stance_label"] = [t[0] for t in art_results]
    df["article_stance_score"] = [t[1] for t in art_results]

    # Summary stance
    if sum_topic_col in df.columns:
        sum_results = df.apply(
            lambda r: classify_stance(stance_clf, r[sum_col], r[sum_topic_col]),
            axis=1,
        )
    else:
        sum_results = df[sum_col].apply(lambda x: classify_stance(stance_clf, x, ""))

    df["summary_stance_label"] = [t[0] for t in sum_results]
    df["summary_stance_score"] = [t[1] for t in sum_results]

    # Stance shift (numeric): summary − article
    df["stance_shift"] = df["summary_stance_score"] - df["article_stance_score"]

    # Quick agreement flag
    df["stance_agreement"] = (df["summary_stance_label"] == df["article_stance_label"])

    # Save
    # output_path = "./tests/output/external_model_sent_baseline.csv"
    output_path = "./tests/output/english_external_model_sent_baseline.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved stance labels/scores and shifts to {output_path}")


    # df = pd.read_csv("./tests/output/external_model_sent_baseline.csv")
    df = pd.read_csv("./tests/output/english_external_model_sent_baseline.csv")
    cols = ["score", "stance_shift"]
    df_article = df.groupby(["Article"], as_index=False)[cols].mean()

    print("Pearson:\n", corr_table_with_pvals(df_article, cols, "pearson"), "\n")
    print("Spearman:\n", corr_table_with_pvals(df_article, cols, "spearman"), "\n")
    print("Kendall:\n", corr_table_with_pvals(df_article, cols, "kendall"), "\n")

    # print("Pearson:\n", corr_table_with_pvals(df, cols, "pearson"), "\n")
    # print("Spearman:\n", corr_table_with_pvals(df, cols, "spearman"), "\n")
    # print("Kendall:\n", corr_table_with_pvals(df, cols, "kendall"), "\n")