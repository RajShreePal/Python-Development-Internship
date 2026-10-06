"""
House Price Prediction using Linear Regression
----------------------------------------------
Beginner-friendly machine learning project.
Libraries: Pandas, NumPy, Matplotlib, Scikit-learn.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # save charts to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = "train.csv"
OUTPUT_DIR = "outputs"

# Features chosen after inspecting the dataset (see README for the reasons).
NUMERIC_FEATURES = [
    "GrLivArea",     # living area (sq ft)
    "LotArea",       # lot size (sq ft)
    "TotalBsmtSF",   # basement area (sq ft)
    "OverallQual",   # overall material/finish quality (1-10)
    "YearBuilt",     # year the house was built
    "BedroomAbvGr",  # bedrooms above ground
    "FullBath",      # full bathrooms
    "HalfBath",      # half bathrooms
    "TotRmsAbvGrd",  # total rooms above ground
    "GarageCars",    # garage capacity (cars)
    "LotFrontage",   # street frontage (has missing values)
]
CATEGORICAL_FEATURES = [
    "Neighborhood",  # location
    "BldgType",      # building type
    "KitchenQual",   # kitchen quality
    "CentralAir",    # central air conditioning (Y/N)
]


def load_data(path):
    """Load the CSV file; stop with a clear message if something is wrong."""
    if not os.path.exists(path):
        sys.exit(f"Error: '{path}' was not found. Place train.csv in the "
                 "same folder as this script.")
    try:
        df = pd.read_csv(path)
    except Exception as error:
        sys.exit(f"Error: could not read '{path}': {error}")
    if df.empty:
        sys.exit("Error: the dataset is empty.")
    return df


def find_target_column(df):
    """Identify the price column (SalePrice, or any column containing 'price')."""
    if "SalePrice" in df.columns:
        return "SalePrice"
    for col in df.columns:
        if "price" in col.lower():
            return col
    sys.exit("Error: could not identify the house price (target) column.")


def explore_data(df, target):
    """Print basic information about the dataset."""
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(exclude="number").columns

    print("First 5 rows (selected columns):")
    print(df[["Id", "Neighborhood", "GrLivArea", "OverallQual", target]].head())
    print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target column: {target}")
    print(f"Numerical columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    print("\nColumn names:")
    print(list(df.columns))
    print("\nData types (count per type):")
    print(df.dtypes.astype(str).value_counts())

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\nColumns with missing values: {len(missing)}")
    print(missing)

    print(f"\nStatistics for {target}:")
    print(df[target].describe())


def preprocess_data(df, target):
    """Select features, fill missing values, encode categories, split X and y."""
    needed = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [target]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        sys.exit(f"Error: expected columns not found in dataset: {absent}")

    data = df[needed].copy()
    data = data.dropna(subset=[target])  # a row without a price is unusable

    # Numerical missing values -> median
    for col in NUMERIC_FEATURES:
        data[col] = data[col].fillna(data[col].median())
    # Categorical missing values -> most frequent value
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].fillna(data[col].mode()[0])

    y = data[target]
    X = data.drop(columns=[target])  # target is NOT part of X

    # Convert text categories to 0/1 columns
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True,
                       dtype=int)

    print(f"\nFeatures after encoding: {X.shape[1]} columns, {X.shape[0]} rows")
    print(f"Missing values left in X: {int(X.isnull().sum().sum())}")
    return X, y


def train_model(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_test, y_pred):
    """Print MAE, MSE, RMSE and R2."""
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation")
    print("----------------")
    print(f"Mean Absolute Error: {mae:,.2f}")
    print(f"Mean Squared Error: {mse:,.2f}")
    print(f"Root Mean Squared Error: {rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")


def show_sample_predictions(y_test, y_pred, n=8):
    """Print a few actual vs predicted prices from the test set."""
    print("\nSample Predictions (test data)")
    print(f"{'Actual Price':>15} {'Predicted Price':>17}")
    for actual, predicted in list(zip(y_test, y_pred))[:n]:
        print(f"{actual:>15,.0f} {predicted:>17,.0f}")


def create_visualizations(y, y_test, y_pred):
    """Save the price distribution and actual-vs-predicted charts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="steelblue",
                label="Predictions")
    low, high = y_test.min(), y_test.max()
    plt.plot([low, high], [low, high], color="red", linestyle="--",
             label="Perfect prediction")
    plt.title("Actual vs Predicted House Prices")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    # 2. Price distribution
    plt.figure(figsize=(8, 6))
    plt.hist(y, bins=40, color="seagreen", edgecolor="white")
    plt.title("Distribution of House Prices")
    plt.xlabel("Sale Price ($)")
    plt.ylabel("Number of Houses")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "price_distribution.png"), dpi=150)
    plt.close()

    print(f"\nCharts saved in the '{OUTPUT_DIR}' folder.")


def main():
    df = load_data(DATA_FILE)
    target = find_target_column(df)
    explore_data(df, target)

    X, y = preprocess_data(df, target)

    # 80% training, 20% testing, fixed random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluate_model(y_test, y_pred)
    show_sample_predictions(y_test, y_pred)
    create_visualizations(y, y_test, y_pred)


if __name__ == "__main__":
    main()
