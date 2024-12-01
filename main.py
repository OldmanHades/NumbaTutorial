import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
import time


# Numba-accelerated function for summing amounts by category
@jit(nopython=True)
def compute_category_sums(amounts, categories, n_categories):
    category_sums = np.zeros(n_categories)

    # Sum amounts for each category
    for i in range(len(amounts)):
        category_idx = categories[i]
        category_sums[category_idx] = category_sums[category_idx] + amounts[i]

    return category_sums


def main():
    # Read parquet file using pandas (Numba doesn't work directly with parquet)
    print("Reading data...")
    df = pd.read_parquet(r"D:\Pythonstuff\polars\transactions.parquet")

    # Convert categorical data to numeric indices for Numba
    print("Processing data...")
    unique_exp_types = df["EXP_TYPE"].unique()
    category_mapping = {cat: idx for idx, cat in enumerate(unique_exp_types)}
    category_indices = np.array(
        [category_mapping[cat] for cat in df["EXP_TYPE"]], dtype=np.int64
    )

    # Convert amounts to numpy array
    amounts = df["AMOUNT"].to_numpy()

    # Time the computation
    start_time = time.time()

    # Compute sums using Numba (pass number of categories instead of unique categories)
    category_sums = compute_category_sums(
        amounts, category_indices, len(unique_exp_types)
    )

    end_time = time.time()
    print(f"Computation time: {end_time - start_time:.4f} seconds")

    # Sort results by total expense
    sorted_indices = np.argsort(-category_sums)  # Negative for descending order
    sorted_exp_types = unique_exp_types[sorted_indices]
    sorted_amounts = category_sums[sorted_indices]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_exp_types)), sorted_amounts)
    plt.xticks(range(len(sorted_exp_types)), sorted_exp_types, rotation=45, ha="right")
    plt.title("Total Expenses by Category")
    plt.xlabel("Expense Type")
    plt.ylabel("Total Amount")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Print some additional statistics (equivalent to the commented queries)
    print("\nAdditional Statistics:")
    print(f"Unique Customer Count: {df['CUST_ID'].nunique()}")

    housing_2020 = df[(df["YEAR"] == 2020) & (df["EXP_TYPE"] == "Housing")][
        "AMOUNT"
    ].sum()
    print(f"2020 Housing Expenses: {int(housing_2020)}")


if __name__ == "__main__":
    main()
