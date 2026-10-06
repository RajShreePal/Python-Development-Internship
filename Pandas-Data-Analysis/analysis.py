"""
Titanic Data Analysis and Visualization
---------------------------------------
Uses Pandas for analysis and Matplotlib/Seaborn for charts.
Reads train.csv, cleans it, calculates statistics, saves three charts
to the outputs/ folder and prints insights calculated from the data.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths: always relative to this script, so it works from any terminal folder
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "train.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

SURVIVED_COLOR = "#2a9d8f"
DIED_COLOR = "#e76f51"


def print_header(title):
    """Print a clear section title in the terminal."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
def load_data(path):
    """Load the CSV file. Stop with a friendly message if it is missing."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: Could not find '{path}'.")
        print("Download train.csv from Kaggle (Titanic competition) and place it")
        print("in the same folder as analysis.py, then run the script again.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("ERROR: train.csv is empty. Please download the file again.")
        sys.exit(1)
    return df


def explore_data(df):
    """Show the basic structure of the dataset."""
    print_header("1. DATASET OVERVIEW")
    print("\nFirst 5 rows:")
    print(df.head())
    print(f"\nNumber of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumn names and data types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())


# ---------------------------------------------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess_data(df):
    """Handle missing values without deleting rows. Returns a cleaned copy."""
    print_header("2. DATA PREPROCESSING")
    clean_df = df.copy()  # keep the original data untouched

    # Age is missing for roughly 20% of passengers. Deleting those rows would
    # throw away a lot of data, so we fill with the MEDIAN age. The median is
    # used instead of the mean because it is not pulled by very old/young ages.
    median_age = clean_df["Age"].median()
    clean_df["Age"] = clean_df["Age"].fillna(median_age)
    print(f"Age: filled missing values with the median age ({median_age:.1f}).")

    # Embarked has only a couple of missing values. It is categorical, so we
    # fill with the MODE (the most common port).
    if clean_df["Embarked"].isnull().any():
        most_common_port = clean_df["Embarked"].mode()[0]
        clean_df["Embarked"] = clean_df["Embarked"].fillna(most_common_port)
        print(f"Embarked: filled missing values with the most common port ({most_common_port}).")

    # Cabin is missing for most passengers, so it cannot be filled reliably.
    # We drop only this COLUMN (not any rows). It is not needed in this project.
    if "Cabin" in clean_df.columns:
        missing_percent = clean_df["Cabin"].isnull().mean() * 100
        clean_df = clean_df.drop(columns=["Cabin"])
        print(f"Cabin: dropped column because {missing_percent:.1f}% of values are missing.")

    print("\nMissing values after cleaning:")
    print(clean_df.isnull().sum())
    return clean_df


# ---------------------------------------------------------------------------
# 3. BASIC DATA ANALYSIS
# ---------------------------------------------------------------------------
def analyze_data(raw_df, clean_df):
    """Calculate summary statistics and return the ones needed for insights."""
    print_header("3. BASIC DATA ANALYSIS")

    # Average age uses the ORIGINAL data so that filled-in values do not
    # influence the result.
    average_age = raw_df["Age"].mean()
    average_fare = clean_df["Fare"].mean()
    total_passengers = len(clean_df)
    survivors = int(clean_df["Survived"].sum())
    overall_survival_pct = clean_df["Survived"].mean() * 100

    # Mean of a 0/1 column = proportion of 1s, so multiplying by 100 gives %.
    survival_by_class = clean_df.groupby("Pclass")["Survived"].mean() * 100
    survival_by_gender = clean_df.groupby("Sex")["Survived"].mean() * 100

    print(f"Average passenger age (known ages only): {average_age:.2f}")
    print(f"Average fare: {average_fare:.2f}")
    print(f"Total passengers: {total_passengers}")
    print(f"Number of survivors: {survivors}")
    print(f"Overall survival percentage: {overall_survival_pct:.2f}%")

    print("\nSurvival percentage by passenger class:")
    for pclass, pct in survival_by_class.items():
        print(f"  Class {pclass}: {pct:.2f}%")

    print("\nSurvival percentage by gender:")
    for sex, pct in survival_by_gender.items():
        print(f"  {sex.capitalize()}: {pct:.2f}%")

    return {
        "overall_survival_pct": overall_survival_pct,
        "survival_by_class": survival_by_class,
        "survival_by_gender": survival_by_gender,
    }


# ---------------------------------------------------------------------------
# 4. VISUALIZATIONS
# ---------------------------------------------------------------------------
def plot_survival_by_class(survival_by_class):
    """Bar chart: survival rate (%) for each passenger class."""
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [f"Class {c}" for c in survival_by_class.index],
        survival_by_class.values,
        color=["#264653", "#2a9d8f", "#e9c46a"],
    )
    # Write the percentage above each bar so the chart is easy to read.
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", fontweight="bold")
    ax.set_title("Survival Rate by Passenger Class", fontsize=14, fontweight="bold")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "survival_by_class.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_age_fare_scatter(raw_df):
    """Scatter plot: Age vs Fare, coloured by survival status."""
    # Only use passengers whose age is really known. Using filled-in ages
    # would create a misleading straight line of points at the median age.
    known = raw_df.dropna(subset=["Age", "Fare"])

    fig, ax = plt.subplots(figsize=(9, 6))
    for status, label, color in [(0, "Did not survive", DIED_COLOR),
                                 (1, "Survived", SURVIVED_COLOR)]:
        group = known[known["Survived"] == status]
        ax.scatter(group["Age"], group["Fare"], label=label,
                   color=color, alpha=0.6, edgecolor="white", s=45)
    ax.set_title("Age vs Fare by Survival Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Fare")
    ax.legend(title="Outcome")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "age_fare_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_correlation_heatmap(clean_df):
    """Heatmap of correlations between the numerical columns."""
    numeric_columns = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    correlation = clean_df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Columns", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return correlation


# ---------------------------------------------------------------------------
# 5. INSIGHTS (all numbers are calculated from the data)
# ---------------------------------------------------------------------------
def print_insights(raw_df, clean_df, stats, correlation):
    print_header("5. INSIGHTS AND OBSERVATIONS")

    # Overall survival
    overall = stats["overall_survival_pct"]
    verdict = "most passengers did not survive" if overall < 50 else "most passengers survived"
    print(f"1. Overall, {overall:.1f}% of passengers survived, so {verdict}.")

    # Class
    by_class = stats["survival_by_class"]
    best_class, worst_class = by_class.idxmax(), by_class.idxmin()
    print(f"2. Class {best_class} had the highest survival rate ({by_class.max():.1f}%) "
          f"and Class {worst_class} the lowest ({by_class.min():.1f}%).")

    # Gender
    by_gender = stats["survival_by_gender"]
    best_gender, worst_gender = by_gender.idxmax(), by_gender.idxmin()
    print(f"3. {best_gender.capitalize()} passengers survived more often "
          f"({by_gender.max():.1f}%) than {worst_gender} passengers ({by_gender.min():.1f}%).")

    # Heatmap: strongest relationship with survival (ignore Survived itself)
    with_survival = correlation["Survived"].drop("Survived")
    strongest = with_survival.abs().idxmax()
    value = with_survival[strongest]
    direction = "positive" if value > 0 else "negative"
    print(f"4. Heatmap: {strongest} has the strongest correlation with Survived "
          f"({value:.2f}, {direction}).")
    if strongest == "Pclass":
        print("   A negative Pclass value means lower class numbers (1st class) "
              "were linked to higher survival.")
    print(f"   Pclass vs Fare correlation is {correlation.loc['Pclass', 'Fare']:.2f} "
          "(negative means higher classes paid higher fares).")

    # Age vs Fare scatter patterns (known ages only)
    known = raw_df.dropna(subset=["Age", "Fare"])
    age_fare_corr = known["Age"].corr(known["Fare"])
    survivor_fare = known.loc[known["Survived"] == 1, "Fare"].mean()
    victim_fare = known.loc[known["Survived"] == 0, "Fare"].mean()
    strength = "weak" if abs(age_fare_corr) < 0.3 else "moderate or strong"
    print(f"5. Age vs Fare: correlation is {age_fare_corr:.2f} ({strength} relationship).")
    print(f"   Average fare: survivors {survivor_fare:.2f} vs non-survivors {victim_fare:.2f}.")
    print(f"   Highest fare paid: {known['Fare'].max():.2f}; the median fare is "
          f"{known['Fare'].median():.2f}, so a few very high fares stand out as outliers.")

    # Age of survivors
    survivor_age = known.loc[known["Survived"] == 1, "Age"].median()
    victim_age = known.loc[known["Survived"] == 0, "Age"].median()
    print(f"6. Median age: survivors {survivor_age:.1f} vs non-survivors {victim_age:.1f}.")


# ---------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_df = load_data(CSV_PATH)
    explore_data(raw_df)

    clean_df = preprocess_data(raw_df)
    stats = analyze_data(raw_df, clean_df)

    print_header("4. CREATING VISUALIZATIONS")
    plot_survival_by_class(stats["survival_by_class"])
    plot_age_fare_scatter(raw_df)
    correlation = plot_correlation_heatmap(clean_df)

    print_insights(raw_df, clean_df, stats, correlation)
    print("\nDone! Charts are saved in the 'outputs' folder.")


if __name__ == "__main__":
    main()
