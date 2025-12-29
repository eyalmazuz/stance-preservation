import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score


if __name__ == "__main__":
    path = "Data/datasets/all_data_with_gpt_labels_clean.csv"
    df = pd.read_csv(path)

    # # Define the columns to check
    # columns_to_check = [
    #     "summary topic gpt", 
    #     "summary stance gpt", 
    #     "article topic gpt", 
    #     "article stance gpt"
    # ]

    # for col in columns_to_check:
    #     df[col] = df[col].astype(str).str.strip()

    # # Remove rows where any of those columns is "None" (string)
    # df_clean = df[~df[columns_to_check].isin(["None", "", "nan"]).any(axis=1)]

    # # Save to new file
    # output_path = path.replace(".csv", "_clean.csv")
    # df_clean.to_csv(output_path, index=False)
    # print("Data cleaned and saved to", output_path)

    # Strip whitespace just in case
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    # Define pairs of manual ↔ GPT labels
    pairs = [
        ("summary topic", "summary topic gpt"),
        ("article topic", "article topic gpt"),
        ("summary stance", "summary stance gpt"),
        ("article stance", "article stance gpt"),
    ]

    # Compare each pair
    text_file_path = "Data/gpt_labels_exploration.txt"
    with open(text_file_path, "w") as text_file:
        text_file.write("GPT Labels Exploration\n")
        text_file.write("=" * 30 + "\n\n")

        for manual, gpt in pairs:
            # print(f"\n🔍 Comparing: {manual} ↔ {gpt}")
            labels = sorted(set(df[manual]) | set(df[gpt]))
            # print("Labels:", labels)
            text_file.write(f"\n🔍 Comparing: {manual} ↔ {gpt}\n")
            text_file.write("Labels: " + ", ".join(labels) + "\n")
            
            text_file.write("\nConfusion Matrix:\n")
            cm = confusion_matrix(df[manual], df[gpt], labels=labels)
            text_file.write(str(cm) + "\n")

            # print("\nConfusion Matrix:")
            # print(confusion_matrix(df[manual], df[gpt], labels=labels))

            text_file.write("\nClassification Report:\n")
            report = classification_report(df[manual], df[gpt], labels=labels, zero_division=0)
            text_file.write(report + "\n")

            # print("\nClassification Report:")
            # print(classification_report(df[manual], df[gpt], labels=labels, zero_division=0))

            text_file.write("Cohen's Kappa: " + str(cohen_kappa_score(df[manual], df[gpt])) + "\n")

            # print("Cohen's Kappa:", cohen_kappa_score(df[manual], df[gpt]))
            text_file.write("=" * 30 + "\n\n")

    print("Exploration complete. Results saved to", text_file_path)
