import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the normalized dataset
df = pd.read_csv("cleaned_featured_retail_data.csv")

# 🛠 Ensure Proper InvoiceDate Parsing
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# ✅ Extract Correct Month, Day, Hour (Excluding Year)
df['InvoiceMonth'] = df['InvoiceDate'].dt.month.astype("Int64")
df['InvoiceDay'] = df['InvoiceDate'].dt.day.astype("Int64")
df['InvoiceHour'] = df['InvoiceDate'].dt.hour.astype("Int64")

# 📊 Remove Outliers (Above 95th Percentile)
for col in ['Quantity', 'TotalPrice']:
    upper_limit = df[col].quantile(0.95)  # 95th percentile
    df = df[df[col] <= upper_limit]

# 🎨 Plot Feature Distributions
features = ["Invoice", "Quantity", "Price", "Customer ID", "TotalPrice",
            "InvoiceMonth", "InvoiceDay", "InvoiceHour"]  # Removed InvoiceYear

num_features = len(features)
num_rows = (num_features // 3) + (num_features % 3 > 0)  # Dynamic row count

fig, axes = plt.subplots(num_rows, 3, figsize=(12, 8))  # Adjusted subplot grid
fig.suptitle("Feature Distributions")

axes = axes.flatten()  # Flatten for easier iteration

for i, feature in enumerate(features):
    if feature in df.columns:
        axes[i].hist(df[feature].dropna(), bins=30, edgecolor="black")
        axes[i].set_title(feature)

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ✅ Save the Final Cleaned Dataset
df.to_csv("final_cleaned_retail_data.csv", index=False)
print("✅ Final cleaned dataset saved as: final_cleaned_retail_data.csv")
