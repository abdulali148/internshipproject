import pandas as pd

# Load the dataset
df = pd.read_csv("cleaned_featured_retail_data.csv")  # Update with the correct file path

# Ensure correct column names
df = df[['Invoice', 'Description']].drop_duplicates()

# Remove missing values
df.dropna(subset=['Invoice', 'Description'], inplace=True)

# Convert Invoice to string (to ensure proper grouping)
df['Invoice'] = df['Invoice'].astype(str)

# Group transactions: Create a basket format for Apriori
basket = df.groupby(['Invoice'])['Description'].apply(list).reset_index()

# Save the prepared data
basket.to_csv("apriori_ready_data.csv", index=False)

print("Data prepared successfully! Saved as 'apriori_ready_data.csv'.")
