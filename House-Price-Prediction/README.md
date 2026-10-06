# House Price Prediction Using Linear Regression

## Objective
This project predicts house prices using **Linear Regression** and demonstrates a basic, complete machine-learning workflow: loading data, exploring it, cleaning it, training a model, evaluating it and visualising the results.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Dataset
- **Source:** Kaggle – *House Prices: Advanced Regression Techniques* (Ames, Iowa housing data), file `train.csv`.
- **Records:** 1,460 houses, 81 columns (38 numerical, 43 categorical).
- **Target variable:** `SalePrice` (sale price in US dollars). Mean ≈ $180,921, median $163,000, minimum $34,900, maximum $755,000.
- **Missing values:** 19 columns contain missing values (for example `LotFrontage` has 259).
- **Features used (15):**
  - Numerical: `GrLivArea`, `LotArea`, `TotalBsmtSF`, `OverallQual`, `YearBuilt`, `BedroomAbvGr`, `FullBath`, `HalfBath`, `TotRmsAbvGrd`, `GarageCars`, `LotFrontage`
  - Categorical: `Neighborhood`, `BldgType`, `KitchenQual`, `CentralAir`

### Why these features?
- **Size:** `GrLivArea`, `LotArea` and `TotalBsmtSF` describe how big the house and lot are, which strongly affects price.
- **Rooms:** `BedroomAbvGr`, `FullBath`, `HalfBath` and `TotRmsAbvGrd` cover the bedroom, bathroom and room counts.
- **Location:** `Neighborhood` captures the area the house is in.
- **Quality, age and extras:** `OverallQual`, `YearBuilt`, `KitchenQual`, `GarageCars`, `BldgType` and `CentralAir` are common price drivers.
- The `Id` column was excluded because it is only an identifier. Columns with a very large share of missing values (such as `PoolQC`, `Alley`, `Fence`) were not used to keep the project simple.

## Project Workflow
```text
Data Collection
      ↓
Data Exploration
      ↓
Data Cleaning
      ↓
Feature Selection
      ↓
Categorical Encoding
      ↓
Train-Test Split
      ↓
Linear Regression
      ↓
Prediction
      ↓
Model Evaluation
      ↓
Visualization
```

## Data Preprocessing
- **Numerical missing values** are filled with the **median** of that column (e.g. `LotFrontage`).
- **Categorical missing values** would be filled with the **most frequent value** (none of the selected categorical columns actually had missing values).
- **Categorical encoding** uses `pd.get_dummies(drop_first=True)`, which turns each category into a 0/1 column. After encoding there are 43 input columns.
- The target `SalePrice` is separated into `y` and is **not** included in `X`.
- Data is split **80% training / 20% testing** (1,168 / 292 rows) with `random_state=42`.

## Model
**Linear Regression** was chosen because the task asks for it, it is simple and easy to explain, and it works well as a baseline for predicting a continuous value such as price. It learns one coefficient per feature and combines them into a price estimate.

## Evaluation Metrics
- **MAE (Mean Absolute Error):** the average size of the prediction error in dollars. Lower is better.
- **MSE (Mean Squared Error):** the average of the squared errors. It penalises large errors heavily. Its unit is dollars squared.
- **RMSE (Root Mean Squared Error):** the square root of MSE, so it is back in dollars and easier to interpret.
- **R² Score:** the proportion of the variation in prices the model explains. 1.0 is perfect; 0 means no better than predicting the average.

## Results
Actual results from running the project on the test set (292 houses):

```text
Model Evaluation
----------------
Mean Absolute Error: 20,943.58
Mean Squared Error: 1,139,266,116.39
Root Mean Squared Error: 33,753.02
R² Score: 0.8515
```

The model explains about 85% of the variation in house prices, and its predictions are off by roughly $21,000 on average.

## Visualizations
- `outputs/actual_vs_predicted.png` – scatter plot of actual vs predicted prices on the test set. Points close to the red dashed line are accurate predictions. The most expensive houses are underestimated.
- `outputs/price_distribution.png` – histogram of all sale prices. Most houses sell between roughly $100,000 and $250,000, with a long tail of expensive houses.

## How to Install
```bash
pip install -r requirements.txt
```

## How to Run
Place `train.csv` in the same folder as the script, then run:
```bash
python house_price_prediction.py
```
The `outputs` folder is created automatically.

## Project Structure
```text
House_Price_Prediction/
│
├── house_price_prediction.py
├── train.csv
├── requirements.txt
├── README.md
│
└── outputs/
    ├── actual_vs_predicted.png
    └── price_distribution.png
```

## Future Improvements
These have **not** been implemented in this project:
- Trying additional regression algorithms
- Hyperparameter tuning
- Better feature engineering (for example, handling skewed prices and outliers)
- Cross-validation for a more reliable performance estimate
- Interactive visualization

## Conclusion
This project showed the full workflow of a regression task: exploring a real dataset, handling missing values, encoding categorical features, training Linear Regression and judging it with MAE, MSE, RMSE and R². A simple linear model with 15 sensible features reached an R² of 0.85, but it struggles with very expensive houses, which shows where more advanced techniques could help.
