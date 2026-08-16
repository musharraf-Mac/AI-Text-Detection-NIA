"""
build_featured_dataset.py
--------------------------
Applies extract_features() to every row of the cleaned dataset,
saves the featured dataset, and produces distribution comparison
plots (human vs ai) for every engineered feature.

Run from the project root:
    python src/build_featured_dataset_b.py

Inputs:
    data/processed/final_training_dataset_b.csv
        required columns: text, author, domain, gen_source,
                           source_dataset, split

Outputs:
    data/processed/featured_dataset_b.csv
    outputs/figures/<feature>_hist.png   (histogram, human vs ai overlaid)
    outputs/figures/<feature>_box.png    (boxplot, human vs ai side by side)
    outputs/feature_summary_stats.csv    (mean/std/median per class per feature)
"""

import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features, FEATURE_NAMES  # noqa: E402

INPUT_PATH = "data/processed/final_training_dataset_b.csv"
OUTPUT_CSV_PATH = "data/processed/featured_dataset_b.csv"
FIGURES_DIR = "outputs/figures"
SUMMARY_STATS_PATH = "outputs/feature_summary_stats.csv"


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"text", "author"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = df["text"].apply(extract_features)
    feats_df = pd.DataFrame(list(feats))
    return pd.concat([df.reset_index(drop=True), feats_df], axis=1)


def plot_feature_distributions(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    human = df[df["author"] == "human"]
    ai = df[df["author"] == "ai"]

    for feature in FEATURE_NAMES:
        # --- Histogram (overlaid) ---
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(human[feature], bins=40, alpha=0.55, label="human", density=True)
        ax.hist(ai[feature], bins=40, alpha=0.55, label="ai", density=True)
        ax.set_title(f"Distribution of {feature}: human vs ai")
        ax.set_xlabel(feature)
        ax.set_ylabel("density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{feature}_hist.png"), dpi=150)
        plt.close(fig)

        # --- Boxplot (side by side) ---
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.boxplot(
            [human[feature].dropna(), ai[feature].dropna()],
            tick_labels=["human", "ai"],
            showfliers=False,
        )
        ax.set_title(f"{feature}: human vs ai")
        ax.set_ylabel(feature)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{feature}_box.png"), dpi=150)
        plt.close(fig)


def summarize_by_class(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("author")[FEATURE_NAMES]
        .agg(["mean", "std", "median"])
    )
    return summary


def main():
    print(f"Loading dataset from {INPUT_PATH} ...")
    df = load_dataset(INPUT_PATH)
    print(f"Loaded {len(df)} rows.")

    print("Extracting features for every row (this may take a moment)...")
    featured_df = build_features(df)

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    featured_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved featured dataset -> {OUTPUT_CSV_PATH} ({len(featured_df)} rows)")

    print("Generating distribution comparison plots...")
    plot_feature_distributions(featured_df, FIGURES_DIR)
    print(f"Saved plots -> {FIGURES_DIR}/")

    print("Computing summary statistics by class...")
    summary = summarize_by_class(featured_df)
    os.makedirs(os.path.dirname(SUMMARY_STATS_PATH), exist_ok=True)
    summary.to_csv(SUMMARY_STATS_PATH)
    print(f"Saved summary stats -> {SUMMARY_STATS_PATH}")

    print("\n=== Feature means by class ===")
    print(summary.xs("mean", axis=1, level=1).round(4))

    print("\nDone.")


if __name__ == "__main__":
    main()
