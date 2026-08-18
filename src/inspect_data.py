"""
MLB Sports Analytics Project
Inspect the raw MLB dataset.
"""

import pandas as pd


def main():
    file_path = "data/raw/NYY_2024.csv"

    df = pd.read_csv(file_path)

    print("\n===== DATASET SHAPE =====")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== FIRST 10 ROWS =====")
    print(df.head(10).to_string())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== RESULT COUNTS =====")
    print(df["W/L"].value_counts(dropna=False))


if __name__ == "__main__":
    main()