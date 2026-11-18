
import pandas as pd
from pathlib import Path

# Find the project folder
project_path = Path(__file__).resolve().parents[1]

# Locate the dataset
file_path = project_path / "data" / "creditcard.csv"

# Load dataset
df = pd.read_csv(file_path)

print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())
# Count genuine and fraudulent transactions

print("\nTransaction Class Distribution:")

print(df["Class"].value_counts())


class_percentage = df["Class"].value_counts(normalize=True) * 100

print("\nTransaction Percentage:")
print(class_percentage)
df["Class"].value_counts()




import matplotlib.pyplot as plt
import seaborn as sns

# Count genuine and fraudulent transactions
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Class"
)

plt.title("Genuine vs Fraud Transactions")
plt.xlabel("Transaction Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.show()


# Analyze transaction amount distribution

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Amount",
    bins=50,
    color="blue"
)

plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.show()



plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.title("Transaction Amount: Genuine vs Fraud")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Transaction Amount")

plt.show()



# Transaction amount distribution

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Amount",
    bins=50
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.show()


plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.title("Transaction Amount: Genuine vs Fraud")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Amount")

plt.show()


# Outlier analysis for transaction Amount

Q1 = df["Amount"].quantile(0.25)
Q3 = df["Amount"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["Amount"] < lower_limit) |
    (df["Amount"] > upper_limit)
]

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)
print("Potential outliers:", len(outliers))


# Correlation analysis

correlation = df.corr()

print("\nCorrelation with Fraud Class:")

print(
    correlation["Class"]
    .sort_values(ascending=False)
)


plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0
)

plt.title("Transaction Feature Correlation Heatmap")
plt.show()