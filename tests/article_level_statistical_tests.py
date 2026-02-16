"""
Overall ranking across models

→ Friedman test
→ Nemenyi post-hoc test
"""
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
from autorank import autorank, plot_stats, create_report
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# --------------------------------------------------------------------------------------------------------------------------------------------
# Friedman
# --------------------------------------------------------------------------------------------------------------------------------------------
# ---- Load merged data ----
# df = pd.read_csv("./tests/test_data/correlations_article_level.csv")
# df = pd.read_csv("./tests/test_data/art_similarity_baselines_correlations.csv")
df = pd.read_csv("./tests/test_data/art_metric_correlations_groupkfold_FIXEDFOLDS_WIDE.csv")

print("DataFrame:\n", df.set_index('Model'))
mean_rank = (
    df.set_index("Model")
      .rank(ascending=True)
      .mean(axis=1)
      .sort_values()
)

print("\nMean Rank (higher = better):\n", mean_rank)

# print("\nRank:\n", df.set_index('Model').rank())

# ---- Friedman test ----
X = df.drop(columns=["Model"]).apply(pd.to_numeric, errors="coerce")

# keep only folds where ALL models have values
X_complete = X.dropna(axis=1, how="any")

print("Original shape:", X.shape)
print("Complete shape:", X_complete.shape) 

stat, p = stats.friedmanchisquare(*[X_complete.iloc[i].to_numpy() for i in range(X_complete.shape[0])])
print("\nFriedman statistic:", stat)
print("Friedman p-value:", p)


data_long = df.melt(id_vars=['Model'])

# Pairwise Wilcoxon signed-rank test with Holm correction
posthoc = sp.posthoc_dunn(data_long, val_col='value', group_col='Model')

print("\n", posthoc)

print(posthoc.loc["emd_score"].drop("emd_score"))

# # Prepare pairwise comparisons
# pairs = [(i, j) for i in range(len(model_cols)) for j in range(i+1, len(model_cols))]
# pvals = []

# for i, j in pairs:
#     x = df[model_cols[i]]
#     y = df[model_cols[j]]

#     # Handle zero-difference case
#     if (x != y).any():
#         stat, p = wilcoxon(x, y)
#     else:
#         p = 1.0  # No difference
#     pvals.append(p)

# # Holm correction
# reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='holm')

# # Show results
# for k, (i, j) in enumerate(pairs):
#     print(f"{model_cols[i]} vs {model_cols[j]}: raw p={pvals[k]:.3e}, Holm-corrected p={pvals_corrected[k]:.3e}, reject={reject[k]}")

# # --- Count wins per model ---
# wins = {model: 0 for model in model_cols}

# for k, (i, j) in enumerate(pairs):
#     if reject[k]:  # significant difference
#         # higher median wins
#         median_i = df[model_cols[i]].median()
#         median_j = df[model_cols[j]].median()
#         if median_i > median_j:
#             wins[model_cols[i]] += 1
#         else:
#             wins[model_cols[j]] += 1

# # --- Rank models by wins ---
# ranking = sorted(wins.items(), key=lambda x: x[1], reverse=True)
# print("\nModel ranking based on pairwise Wilcoxon + Holm correction:")
# for rank, (model, score) in enumerate(ranking, 1):
#     print(f"{rank}. {model} (wins: {score})")