"""NumPy Matrix Operations Tool

A command-line program that performs basic matrix operations with NumPy:
addition, subtraction, multiplication, transpose and determinant.
"""

import numpy as np

LINE = "=" * 40


def print_menu():
    print()
    print(LINE)
    print("      NUMPY MATRIX OPERATIONS TOOL")
    print(LINE)
    print()
    print("1. Matrix Addition")
    print("2. Matrix Subtraction")
    print("3. Matrix Multiplication")
    print("4. Matrix Transpose")
    print("5. Matrix Determinant")
    print("6. Exit")
    print()


def read_positive_int(prompt):
    """Keep asking until the user enters a whole number greater than 0."""
    while True:
        text = input(prompt).strip()
        try:
            value = int(text)
        except ValueError:
            print("Invalid input. Please enter a whole number (e.g. 2).")
            continue
        if value < 1:
            print("The value must be at least 1.")
            continue
        return value


def parse_number(token):
    """Convert text to int if possible, otherwise float. Raises ValueError."""
    try:
        return int(token)
    except ValueError:
        number = float(token)
        if not np.isfinite(number):  # rejects nan and inf
            raise ValueError(f"'{token}' is not a finite number")
        return number


def get_matrix(name="Matrix"):
    """Ask for dimensions and values, and return the matrix as a NumPy array."""
    print(f"\n--- Enter {name} ---")
    rows = read_positive_int("Enter number of rows: ")
    cols = read_positive_int("Enter number of columns: ")
    print(f"Enter {rows} row(s), each with {cols} number(s) separated by spaces.")

    data = []
    for i in range(rows):
        while True:
            line = input(f"Row {i + 1}: ")
            try:
                values = [parse_number(token) for token in line.split()]
            except ValueError:
                print("Invalid input. Please enter numbers only (e.g. 1 2.5 -3).")
                continue
            if len(values) != cols:
                print(f"Please enter exactly {cols} number(s); you entered {len(values)}.")
                continue
            data.append(values)
            break

    return np.array(data)


def display_matrix(matrix, title="Matrix"):
    print(f"\n{title}:")
    print(matrix)


def add_matrices(a, b):
    if a.shape != b.shape:
        raise ValueError(
            f"Addition needs matrices of the same size, but got {a.shape} and {b.shape}."
        )
    return np.add(a, b)


def subtract_matrices(a, b):
    if a.shape != b.shape:
        raise ValueError(
            f"Subtraction needs matrices of the same size, but got {a.shape} and {b.shape}."
        )
    return np.subtract(a, b)


def multiply_matrices(a, b):
    # Columns of A must equal rows of B
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Cannot multiply: columns of A ({a.shape[1]}) must equal rows of B ({b.shape[0]})."
        )
    return np.matmul(a, b)


def transpose_matrix(a):
    return a.T


def determinant_matrix(a):
    if a.shape[0] != a.shape[1]:
        raise ValueError(
            f"Determinant needs a square matrix, but got {a.shape[0]}x{a.shape[1]}."
        )
    result = round(float(np.linalg.det(a)), 4) + 0.0  # "+ 0.0" turns -0.0 into 0.0
    if result == int(result):
        return int(result)
    return result


def run_operation(choice):
    """Collect the needed matrices for the chosen operation and show the result."""
    if choice in ("1", "2", "3"):
        a = get_matrix("Matrix A")
        b = get_matrix("Matrix B")
        display_matrix(a, "Matrix A")
        display_matrix(b, "Matrix B")
        if choice == "1":
            display_matrix(add_matrices(a, b), "Result (A + B)")
        elif choice == "2":
            display_matrix(subtract_matrices(a, b), "Result (A - B)")
        else:
            display_matrix(multiply_matrices(a, b), "Result (A x B)")
    else:
        a = get_matrix("Matrix A")
        display_matrix(a, "Matrix A")
        if choice == "4":
            display_matrix(transpose_matrix(a), "Transpose of Matrix A")
        else:
            print(f"\nDeterminant of Matrix A: {determinant_matrix(a)}")


def main():
    print("Welcome to the NumPy Matrix Operations Tool!")
    try:
        while True:
            print_menu()
            choice = input("Enter your choice: ").strip()
            if choice == "6":
                print("\nThank you for using the tool. Goodbye!")
                break
            if choice not in ("1", "2", "3", "4", "5"):
                print("Invalid choice. Please enter a number from 1 to 6.")
                continue
            try:
                run_operation(choice)
            except ValueError as error:
                print(f"\nError: {error}")
    except (EOFError, KeyboardInterrupt):
        print("\n\nInput ended. Goodbye!")


if __name__ == "__main__":
    main()
