import argparse
from pathlib import Path
import polars as pl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    return parser.parse_args()

def is_missing(val):
    if val is None: return True
    if isinstance(val, float) and val != val: return True
    return str(val).strip().lower() in {"", "nan", "null", "none"}

def main():
    args = parse_args()
    df = pl.read_csv(args.input_csv)
    
    topic_match_stance_match = 0
    topic_match_stance_mismatch = 0
    topic_mismatch_stance_match = 0
    topic_mismatch_stance_mismatch = 0

    for row in df.iter_rows(named=True):
        st = row.get("majority_summary_topic")
        at = row.get("majority_article_topic")
        ss = row.get("majority_summary_stance")
        ast = row.get("majority_article_stance")

        if is_missing(st) or is_missing(at) or is_missing(ss) or is_missing(ast):
            continue

        # Normalized values are already exact string matches in this dataset
        topic_match = (st == at)
        stance_match = (ss == ast)

        if topic_match and stance_match:
            topic_match_stance_match += 1
        elif topic_match and not stance_match:
            topic_match_stance_mismatch += 1
        elif not topic_match and stance_match:
            topic_mismatch_stance_match += 1
        else:
            topic_mismatch_stance_mismatch += 1

    total_topic_match = topic_match_stance_match + topic_match_stance_mismatch
    total_topic_mismatch = topic_mismatch_stance_match + topic_mismatch_stance_mismatch
    total_stance_match = topic_match_stance_match + topic_mismatch_stance_match
    total_stance_mismatch = topic_match_stance_mismatch + topic_mismatch_stance_mismatch
    total = total_topic_match + total_topic_mismatch

    print(f"Matrix for {args.input_csv.name}, {total} rows:")
    print(f"{'':<25} {'Stance match':<15} {'Stance mismatch':<15} {'Total'}")
    print(f"{'Topic match':<25} {topic_match_stance_match:<15} {topic_match_stance_mismatch:<15} {total_topic_match}")
    print(f"{'Topic mismatch':<25} {topic_mismatch_stance_match:<15} {topic_mismatch_stance_mismatch:<15} {total_topic_mismatch}")
    print(f"{'Total':<25} {total_stance_match:<15} {total_stance_mismatch:<15} {total}")
    print()

if __name__ == "__main__":
    main()
