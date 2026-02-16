import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
from scipy.stats import pearsonr, spearmanr, kendalltau


_ = load_dotenv(find_dotenv())

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID")
)


def get_completion(prompt, model="gpt-4o-mini"):
    messages = [
            {"role": "system", "content": "You are a helpful assistant that jugdes the quality of preserving the stance in text summaries."},
            {"role": "user", "content": prompt}
            ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5, 
    )
    return response.choices[0].message.content

def get_completion_from_messages(messages, model="gpt-4o-mini", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, 
    )
    return response.choices[0].message.content

def corr_with_pvalues(df, cols, method="pearson"):
    results = pd.DataFrame(index=cols, columns=cols)

    for i in cols:
        for j in cols:
            if method == "pearson":
                r, p = pearsonr(df[i], df[j])
            elif method == "spearman":
                r, p = spearmanr(df[i], df[j])
            elif method == "kendall":
                r, p = kendalltau(df[i], df[j])
            else:
                raise ValueError("method must be pearson/spearman/kendall")
            
            results.loc[i, j] = f"{r:.3f} (p={p:.3g})"
    return results


if __name__ == "__main__":
    # Load dataset
    # df = pd.read_csv("./tests/output/labeled_data_with_scores.csv")    
    df = pd.read_csv("./tests/output/english_labeled_data_with_scores.csv")

    # sum_col = "Summary"
    # art_col = "Article"
    sum_col = "Sentence in Summary"
    art_col = "Best Match Sentences From Article"

    for index, row in df.iterrows():
        print(f"Processing row {index+1}/{len(df)}")
        article = row[art_col]
        summary = row[sum_col]

        # prompt = f"""
        #     Given the following article and its summary, rate the quality of the stance preservation in the summary on a scale from 1 to 9, where 1 is very poor and 9 is excellent. 
        #     Remember that stance is not sentiment. Stance refers to the position or attitude expressed in the text towards a particular topic or entity.
        #     Take in count the sentences in the summary and their relation to the article. Each sentence in the summary should reflect the stance of the article and then you can compare the stance and calculate some score - 
        #     if the stance is not preserved reduce points from the score.

        #     Article: {article}
        #     Summary: {summary}

        #     Provide only the numeric score. No additional text.
        # """

        prompt = f"""
            Given the following sentence of a summary and its matching sentence from an article, rate the quality of the stance preservation in the summary on a scale from 1 to 9, where 1 is very poor and 9 is excellent. 
            Remember that stance is not sentiment. Stance refers to the position or attitude expressed in the text towards a particular topic or entity.
            Take in count the sentences in the summary and their relation to the article. Each sentence in the summary should reflect the stance of the article and then you can compare the stance and calculate some score - 
            if the stance is not preserved reduce points from the score.

            Article: {article}
            Summary: {summary}

            Provide only the numeric score. No additional text.
        """
        try:
            score = get_completion(prompt)
            print(f"Score: {score}")
            df.at[index, 'LLM_score'] = score
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            df.at[index, 'LLM_score'] = None

    # df.to_csv("./tests/output/labeled_data_with_LLM_sent_scores.csv", index=False)
    df.to_csv("./tests/output/english_labeled_data_with_LLM_sent_scores.csv", index=False)


    # compare scores of LLM with our score
    # df = pd.read_csv("./tests/output/labeled_data_with_LLM_sent_scores.csv")
    df = pd.read_csv("./tests/output/english_labeled_data_with_LLM_sent_scores.csv")
    df["LLM_score"] = df["LLM_score"]/9

    agg_cols = ["sent_score"]
    df_article = df.groupby("Article", as_index=False)[agg_cols].mean()
    df = df.merge(df_article, on="Article", suffixes=("", "_mean"))

    cols = ['score','sent_score_mean', 'LLM_score']

    pearson_results = corr_with_pvalues(df, cols, method="pearson")
    spearman_results = corr_with_pvalues(df, cols, method="spearman")
    kendall_results = corr_with_pvalues(df, cols, method="kendall")

    print("\nPearson correlations (with p-values):\n", pearson_results, "\n")
    print("\nSpearman correlations (with p-values):\n", spearman_results, "\n")
    print("\nKendall correlations (with p-values):\n", kendall_results, "\n")
    
    # correlation = df['score'].corr(df['LLM_score'])
    # print(f"Correlation between human scores and LLM scores: {correlation:.4f}")
