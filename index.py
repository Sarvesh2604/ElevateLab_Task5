# Install if needed
# pip install pandas matplotlib seaborn openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# Load CSV file
df = pd.read_csv("Excel-Formulas-and-Functions-Complete-Sheet.csv")

# Preview data
df.head()

# Shape of dataset
print("Rows:", df.shape[0], " Columns:", df.shape[1])

# Column names
print("Columns:", df.columns.tolist())

# Data types & missing values
df.info()

# Statistical summary (numeric columns)
df.describe()

# Check missing values count
df.isnull().sum()

num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

cat_cols = df.select_dtypes(exclude=np.number).columns
for col in cat_cols:
    plt.figure(figsize=(6,3))
    df[col].value_counts().head(10).plot(kind='bar', color='skyblue')
    plt.title(f"Top Categories in {col}")
    plt.show()

sns.pairplot(df[num_cols])
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Example: Replace 'CategoryColumn' and 'NumericColumn' with actual columns
sns.boxplot(data=df, x='CategoryColumn', y='NumericColumn')
plt.xticks(rotation=45)
plt.show()

for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Fill missing values
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Example findings
print("✅ Dataset has", df.shape[0], "rows and", df.shape[1], "columns.")
print("✅ No severe multicollinearity found.")
print("✅ Most numerical columns are right-skewed.")
print("✅ Some categorical variables have dominant single values.")

df.to_csv("cleaned_dataset.csv", index=False)
