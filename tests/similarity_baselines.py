import pandas as pd
from rouge_score import rouge_scorer
import sacrebleu
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# # Load your labeled dataset
# # df = pd.read_csv("./tests/output/labeled_data_with_scores.csv")
# df = pd.read_csv("./tests/output/english_labeled_data_with_scores.csv")

# scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# rouge1_scores, rouge2_scores, rougeL_scores, bleu_scores = [], [], [], []

# model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# model = SentenceTransformer(model_name)

# # sum_col = "Sentence in Summary"
# # art_col = "Best Match Sentences From Article"
# sum_col = "Summary"
# art_col = "Article"

# summary_embs = model.encode(df[sum_col].tolist(), batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
# article_embs = model.encode(df[art_col].tolist(), batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

# sims = np.sum(summary_embs * article_embs, axis=1)  
# df["EMB_SIM"] = sims


# # Create TF-IDF vectorizer (English example; set analyzer='word' or 'char')
# vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5))

# # Fit on all text (optional: you can fit only on articles)
# vectorizer.fit(df[art_col].tolist() + df[sum_col].tolist())

# # Transform texts into vectors
# article_vecs = vectorizer.transform(df[art_col])
# summary_vecs = vectorizer.transform(df[sum_col])

# # Compute cosine similarity row by row
# similarities = []
# for i in range(len(df)):
#     sim = cosine_similarity(article_vecs[i], summary_vecs[i])[0][0]
#     similarities.append(sim)

# df["TFIDF_COS_SIM"] = similarities

# for _, row in df.iterrows():
#     summary_sentence = str(row[sum_col])
#     article_sentence = str(row[art_col])

#     # ROUGE
#     rouge = scorer.score(article_sentence, summary_sentence)
#     rouge1_scores.append(rouge['rouge1'].fmeasure)
#     rouge2_scores.append(rouge['rouge2'].fmeasure)
#     rougeL_scores.append(rouge['rougeL'].fmeasure)

#     # BLEU
#     bleu = sacrebleu.sentence_bleu(summary_sentence, [article_sentence])
#     bleu_scores.append(bleu.score)

# # Add results to dataframe
# df["ROUGE-1"] = rouge1_scores
# df["ROUGE-2"] = rouge2_scores
# df["ROUGE-L"] = rougeL_scores
# df["BLEU"] = bleu_scores

# # agg_cols = ["sent_score", "EMB_SIM", "TFIDF_COS_SIM", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
# # agg_cols = ["EMB_SIM", "TFIDF_COS_SIM", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
# # df_article = df.groupby("Article", as_index=False)[agg_cols].mean()
# # df = df.merge(df_article, on="Article", suffixes=("", "_sent_mean"))

# # df.to_csv("./tests/output/dataset_with_baselines_sent_mean_score.csv", index=False)
# # df.to_csv("./tests/output/dataset_with_baselines_sent.csv", index=False)
# # df.to_csv("./tests/output/english_dataset_with_baselines_sent.csv", index=False)
# # df.to_csv("./tests/output/dataset_with_baselines.csv", index=False)
# df.to_csv("./tests/output/english_dataset_with_baselines.csv", index=False)



# Load dataset (with your computed scores + ROUGE + BLEU)
# df = pd.read_csv("./tests/output/dataset_with_baselines_sent.csv")
# df = pd.read_csv("./tests/output/english_dataset_with_baselines_sent.csv")
# df = pd.read_csv("./tests/output/dataset_with_baselines_sent_mean_score.csv")
# df = pd.read_csv("./tests/output/dataset_with_baselines.csv")
df = pd.read_csv("./tests/output/english_dataset_with_baselines.csv")

df_match = df[df['summary stance'] == df['article stance']]
df_no_match = df[df['summary stance'] != df['article stance']]
# df_perfect_match = df[df['score'] == 1]
# df_no_match = df[df['score'] < 1]
# df_no_match_sent = df_no_match[df_no_match['summary stance'] != df_no_match['article stance']]
# df_perfect_match_art = df_perfect_match.groupby("Article", as_index=False)
# df_no_match_art = df_no_match.groupby("Article", as_index=False)

# cols = ["sent_score_sent_mean", "EMB_SIM_sent_mean", "TFIDF_COS_SIM_sent_mean", "ROUGE-1_sent_mean", "ROUGE-2_sent_mean", "ROUGE-L_sent_mean", "BLEU_sent_mean"]
# cols = ["score", "EMB_SIM_sent_mean", "TFIDF_COS_SIM_sent_mean", "ROUGE-1_sent_mean", "ROUGE-2_sent_mean", "ROUGE-L_sent_mean", "BLEU_sent_mean"]
# cols = ["sent_score", "EMB_SIM", "TFIDF_COS_SIM", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
cols = ["score", "EMB_SIM", "TFIDF_COS_SIM", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]

# df = df_no_match
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

pearson_results = corr_with_pvalues(df, cols, method="pearson")
spearman_results = corr_with_pvalues(df, cols, method="spearman")
kendall_results = corr_with_pvalues(df, cols, method="kendall")

print("\nPearson correlations (with p-values):\n", pearson_results, "\n")
print("\nSpearman correlations (with p-values):\n", spearman_results, "\n")
print("\nKendall correlations (with p-values):\n", kendall_results, "\n")

