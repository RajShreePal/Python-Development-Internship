# Titanic Data Analysis and Visualization

## Objective

Use Python, Pandas, Matplotlib, and Seaborn to load and analyze the real Kaggle Titanic dataset, handle missing values, calculate summary statistics, create visualizations, and identify meaningful patterns in the data.

## Technologies Used

* **Python 3.8+**
* **Pandas** — data loading, cleaning, and analysis
* **Matplotlib** — data visualization
* **Seaborn** — correlation heatmap

## Dataset

The project uses the **Kaggle Titanic: Machine Learning from Disaster** dataset (`train.csv`). Each row represents one passenger.

| Column   | Meaning                           |
| -------- | --------------------------------- |
| Survived | 0 = did not survive, 1 = survived |
| Pclass   | Passenger class (1st, 2nd, 3rd)   |
| Sex      | Passenger gender                  |
| Age      | Age in years                      |
| SibSp    | Number of siblings/spouses aboard |
| Parch    | Number of parents/children aboard |
| Fare     | Ticket price                      |
| Cabin    | Cabin number                      |
| Embarked | Port of embarkation               |

The dataset is loaded directly from the CSV file using `pandas.read_csv()`.

## Features and Analysis Performed

The project performs the following tasks:

* Loads the Titanic CSV dataset using Pandas
* Displays the first five rows
* Displays the number of rows and columns
* Displays column names and data types
* Identifies missing values
* Handles missing data:

  * Missing `Age` values are replaced with the median age
  * Missing `Embarked` values are replaced with the most common port
  * `Cabin` is removed because approximately 77.1% of its values are missing
* Calculates:

  * Average passenger age
  * Average fare
  * Total number of passengers
  * Number of survivors
  * Overall survival percentage
  * Survival percentage by passenger class
  * Survival percentage by gender
* Creates and saves three visualizations
* Generates data-driven observations from the analysis

## Dataset Overview

The dataset contains:

* **891 passengers**
* **12 columns**
* **177 missing Age values**
* **687 missing Cabin values**
* **2 missing Embarked values**

After preprocessing, the selected dataset columns contain no missing values.

## Visualizations

| Chart                            | File                              |
| -------------------------------- | --------------------------------- |
| Survival rate by passenger class | `outputs/survival_by_class.png`   |
| Age vs Fare by survival status   | `outputs/age_fare_scatter.png`    |
| Correlation heatmap              | `outputs/correlation_heatmap.png` |

### 1. Survival Rate by Passenger Class

The bar chart compares the percentage of passengers who survived in each passenger class.

* Class 1: **62.96%**
* Class 2: **47.28%**
* Class 3: **24.24%**

The visualization shows that survival rate decreased substantially from first class to third class.

### 2. Age vs Fare Scatter Plot

The scatter plot compares passenger age and ticket fare while distinguishing between survivors and non-survivors.

The correlation between Age and Fare is approximately **0.10**, indicating a weak linear relationship.

The analysis also shows:

* Average fare among survivors: **51.84**
* Average fare among non-survivors: **22.97**
* Highest fare: **512.33**

Most passengers paid relatively low fares, while a small number of passengers paid exceptionally high fares.

### 3. Correlation Heatmap

The heatmap shows the correlation between numerical variables in the dataset.

The strongest correlation with `Survived` among the analyzed numerical variables was `Pclass`, with a correlation of approximately **-0.34**.

The negative correlation means that lower numerical passenger-class values, particularly first class, were associated with higher survival rates.

`Pclass` and `Fare` also showed a negative correlation of approximately **-0.55**.

## Key Findings

The analysis produced the following findings:

1. The overall survival rate was **38.38%**, meaning that most passengers in the dataset did not survive.

2. Passenger class had a noticeable relationship with survival:

   * **1st class: 62.96%**
   * **2nd class: 47.28%**
   * **3rd class: 24.24%**

3. Female passengers had a substantially higher survival rate than male passengers:

   * **Female: 74.20%**
   * **Male: 18.89%**

4. `Pclass` had the strongest correlation with `Survived` among the selected numerical variables, with a correlation of approximately **-0.34**.

5. The Age vs Fare correlation was approximately **0.10**, indicating a weak linear relationship between passenger age and fare.

6. Survivors paid a higher average fare (**51.84**) compared with non-survivors (**22.97**).

7. The analysis demonstrates that passenger class and gender were important factors associated with survival in the Titanic dataset.

## How to Install Dependencies

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

## How to Run

1. Place `train.csv` in the same folder as `analysis.py`.
2. Open a terminal in the project folder.
3. Install the required dependencies.
4. Run the program:

```bash
python analysis.py
```

The program analyzes the dataset and automatically saves the three visualizations inside the `outputs` folder.

## Project Structure

```text
Titanic_Data_Analysis/
│
├── analysis.py
├── README.md
├── requirements.txt
├── train.csv
│
└── outputs/
    ├── survival_by_class.png
    ├── age_fare_scatter.png
    └── correlation_heatmap.png
```

## Example Results

```text
Number of rows: 891
Number of columns: 12

Average passenger age: 29.70
Average fare: 32.20

Total passengers: 891
Number of survivors: 342
Overall survival percentage: 38.38%

Survival percentage by passenger class:
Class 1: 62.96%
Class 2: 47.28%
Class 3: 24.24%

Survival percentage by gender:
Female: 74.20%
Male: 18.89%
```

## Future Improvements

Possible improvements to the project include:

* Add an age-distribution histogram
* Analyze survival by port of embarkation
* Create age groups such as child, adult, and senior
* Compare survival across multiple passenger characteristics
* Export analysis results to CSV or Excel
* Build an interactive dashboard for the analysis

## Conclusion

This project demonstrates the use of Python for practical data analysis. Pandas was used for loading, cleaning, and analyzing the dataset, while Matplotlib and Seaborn were used to visualize patterns and relationships. The analysis shows clear differences in survival rates based on passenger class and gender and provides a practical introduction to exploratory data analysis.
