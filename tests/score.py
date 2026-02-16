# יצירת ציון פקטיבי לכל זוג (מאמר, סיכום) - לכל זוג לבדוק את ההתאמה בין כל זוגות המשפטים ולחשב איזשהו ציון (לדוגמא יש התאמה בין 9 משפטים מתוך 10 אז הציון הוא 9/10) 
# data path: ./Data/datasets/labeled_data.csv

import pandas as pd

# stance similarity mapping
def stance_sim(s1, s2):
    if s1 == s2:
        return 1.0
    if "נייטרלי" in (s1, s2):  # one neutral, one polarized
        return 0.5
    return 0.0  # direct opposition


def calc_soft_score(group):
    same_topic = group[group["summary topic"] == group["article topic"]]
    if len(same_topic) == 0:
        group["score"] = 0
        return group

    sims = same_topic.apply(lambda row: stance_sim(row["summary stance"], row["article stance"]), axis=1)
    score = sims.mean() if len(sims) > 0 else 0
    group["score"] = score
    return group


def calculate_pair_score(group):
    """Calculate score for one article-summary pair"""
    # Only consider sentences where topics match
    matching_topics = group[group["summary topic"] == group["article topic"]]
    
    if len(matching_topics) == 0:
        return 0.0
    
    # Calculate stance similarities for matching topics
    similarities = matching_topics.apply(
        lambda row: stance_sim(row["summary stance"], row["article stance"]), 
        axis=1
    )
    
    return similarities.mean()


def calc_score(group):
        topic_not_matches = (group['summary topic'] != group['article topic']).sum()
        matches = ((group['summary stance'] == group['article stance']) & (group['summary topic'] == group['article topic'])).sum()
        total = len(group)
        score = matches / (total - topic_not_matches) if (total - topic_not_matches) > 0 else 0
        group['score'] = score
        return group

if __name__ == "__main__":
    # path = "./Data/datasets/labeled_data.csv"
    path = "./Data/datasets/english_labeled_data_completed.csv"
    df = pd.read_csv(path)

    df = df.groupby(['Article', 'Summary']).apply(calc_score)
    df['sent_score'] = ((df['summary stance'] == df['article stance']) & (df['summary topic'] == df['article topic'])).astype(int)
    df.to_csv("./tests/output/english_labeled_data_with_scores.csv", index=False)

    # # Calculate pair-level scores
    # pair_scores = df.groupby(['Article', 'Summary']).apply(calculate_pair_score)
    # pair_scores.name = 'score'
    
    # # Merge back to original dataframe
    # df = df.merge(pair_scores.reset_index(), on=['Article', 'Summary'])

    # # per-sentence stance similarity
    # df["sent_score"] = df.apply(
    #     lambda row: stance_sim(row["summary stance"], row["article stance"])
    #     if row["summary topic"] == row["article topic"] else 0.0,
    #     axis=1
    # )

    # df.to_csv("./tests/output/labeled_data_with_soft_scores.csv", index=False)
