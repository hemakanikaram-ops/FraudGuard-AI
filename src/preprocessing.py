import pandas as pd
from pathlib import Path

# Locate project and dataset
project_path = Path(__file__).resolve().parents[1]
file_path = project_path / "data" / "creditcard.csv"

# Load dataset
df = pd.read_csv(file_path)

# Check missing values
print("Missing values:")
print(df.isnull().sum())

# Total missing values
print("\nTotal missing values:")
print(df.isnull().sum().sum())

# Check duplicate rows
print("Duplicate records:", df.duplicated().sum())

print("\nColumn data types:")
print(df.dtypes)