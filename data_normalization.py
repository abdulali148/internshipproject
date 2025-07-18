import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = "cleaned_featured_retail_data.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Identify numerical columns for normalization
num_cols = ['Quantity', 'Price', 'TotalPrice', 'InvoiceHour']

# Apply Min-Max Scaling (0 to 1)
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save the normalized dataset
output_file = "normalized_retail_data.csv"
df.to_csv(output_file, index=False)

print(f"Data normalization complete! Saved as: {output_file}")
