import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau


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
    test_path = "./tests/output/labeled_data_with_scores.csv"
    # test_path = "./tests/output/labeled_data_with_soft_scores.csv"
    # model_output_path = "./scripts/output/test_merged.csv"
    # model_output_path = "./scripts/output/test_entropy_merged.csv"
    # model_output_path = "./scripts/output/test_topic_merged.csv"
    model_output_path = "./scripts/output/fine_tuned_test_merged.csv"
    # model_output_path = "./scripts/output/stance_noentropy_emd_merged.csv"


    # Load CSVs
    test_df = pd.read_csv(test_path)
    model_output_df = pd.read_csv(model_output_path)

    # Group by Article + Summary and compute mean scores
    # test_grouped = test_df.groupby(['Summary'])['score'].mean().reset_index().rename(columns={'score': 'score_test'})
    # test_grouped = test_df.groupby(['Summary'])['sent_score'].mean().reset_index().rename(columns={'sent_score': 'score_test'})
    # test_grouped = test_df.groupby(['Summary'])['sent_score_mean'].mean().reset_index().rename(columns={'sent_score_mean': 'score_test'})
    model_output_grouped = model_output_df.groupby(['Summary'])['emd_score'].mean().reset_index().rename(columns={'emd_score': 'emd_score_model'})

    # Merge on Article and Summary
    # correlation_df = pd.merge(test_grouped, model_output_grouped, on=['Summary'], how="inner")
    correlation_df = pd.merge(test_df, model_output_grouped, on=['Summary'], how="inner")
    # correlation_df = pd.merge(test_df, model_output_df, on=['Summary'], how="inner")

    print("Merged DataFrame shape:", correlation_df.shape)
    print(correlation_df.head())

    # Compute Pearson correlation
    # correlation = correlation_df['score_test'].corr(correlation_df['emd_score_model'])
    # cols = ["score_test", "emd_score_model"]
    cols = ["sent_score", "emd_score_model"]
    # cols = ["sent_score", "emd_score"]

    print("Pearson:\n", corr_table_with_pvals(correlation_df, cols, "pearson"), "\n")
    print("Spearman:\n", corr_table_with_pvals(correlation_df, cols, "spearman"), "\n")
    print("Kendall:\n", corr_table_with_pvals(correlation_df, cols, "kendall"), "\n")
    # print(f"Correlation between test scores and model EMD scores: {correlation:.4f}")


    