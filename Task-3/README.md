# NumPy Matrix Operations Tool

## Project Overview
A command-line application written in Python that lets the user enter matrices and perform common matrix operations. All results are calculated dynamically with NumPy based on the values the user types in.

## Objective
To demonstrate how matrix operations can be performed with NumPy while practising functions, input validation and exception handling in a simple interactive program.

## Features
- Matrix Addition
- Matrix Subtraction
- Matrix Multiplication
- Matrix Transpose
- Matrix Determinant
- Interactive CLI
- Input validation
- Error handling

## Technologies Used
- Python 3
- NumPy

## Operations Explained
- **Addition** (`np.add`): adds matching elements of two matrices. Both matrices must have the same dimensions.
- **Subtraction** (`np.subtract`): subtracts matching elements of the second matrix from the first. Both matrices must have the same dimensions.
- **Multiplication** (`np.matmul`): row-by-column matrix product. The number of columns in A must equal the number of rows in B.
- **Transpose** (`A.T`): flips a matrix over its diagonal, so rows become columns.
- **Determinant** (`np.linalg.det`): a single number calculated from a square matrix. Results are rounded to 4 decimal places.

## Installation
Make sure Python 3 is installed, then install the dependency:

```
pip install -r requirements.txt
```

## How to Run

```
python matrix_operations.py
```

Choose an operation from the menu, then enter the number of rows and columns and the matrix values (one row per line, numbers separated by spaces).

## Example Usage
> This is an **example of usage and expected output**, not output that is hard-coded into the program. The program always calculates results from the values you enter.

```
Matrix A:
[[1 2]
 [3 4]]

Matrix B:
[[5 6]
 [7 8]]

Addition:
[[ 6  8]
 [10 12]]

Subtraction:
[[-4 -4]
 [-4 -4]]

Multiplication:
[[19 22]
 [43 50]]

Transpose of Matrix A:
[[1 3]
 [2 4]]

Determinant of Matrix A:
-2
```

## Project Structure

```
NumPy_Matrix_Operations/
│
├── matrix_operations.py
├── README.md
├── requirements.txt
└── screenshots/
```

## Learning Outcomes
- Working with NumPy arrays
- Performing matrix operations
- Organising code into functions
- Input validation
- Exception handling
- Command-line interaction

## Future Improvements
- Saving matrices and results to a file
- Supporting larger datasets (e.g. loading matrices from CSV)
- Adding a GUI
- Adding more matrix operations (inverse, rank, etc.)

## Conclusion
This project shows how NumPy simplifies matrix calculations and how a small, well-structured command-line program can validate input and handle errors gracefully. It provides a solid foundation for more advanced numerical programs.
