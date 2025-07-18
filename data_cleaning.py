import pandas as pd

# Load the correct cleaned file
file_path = "cleaned_retail_data.csv"

# Read the dataset and debug column names
df = pd.read_csv(file_path)
print("Columns in dataset:", df.columns)  # Debugging step

# Fix column name issue
if 'Quantity' not in df.columns or 'Price' not in df.columns:
    print("Error: 'Quantity' or 'Price' column not found! Check your dataset.")
    exit()

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Feature Engineering
df['TotalPrice'] = df['Quantity'] * df['Price']  # Using 'Price' instead of 'UnitPrice'
df['InvoiceYear'] = df['InvoiceDate'].dt.year
df['InvoiceMonth'] = df['InvoiceDate'].dt.month
df['InvoiceDay'] = df['InvoiceDate'].dt.day
df['InvoiceHour'] = df['InvoiceDate'].dt.hour
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()

# Save the processed file
output_file = "cleaned_featured_retail_data.csv"
df.to_csv(output_file, index=False)

print(f"Feature engineering complete! Saved as: {output_file}")
