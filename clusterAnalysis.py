import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ======= STEP 1: LOAD PROCESSED DATA =======
file_path = "processed_customer_data.csv"  # Ensure correct file path
customer_data = pd.read_csv(file_path)

# Display first few rows to confirm data is loaded
print("First few rows of the dataset:")
print(customer_data.head())

# ======= STEP 2: FEATURE SCALING =======
# Select features to cluster on
features = ['TotalSpent', 'TotalQuantity']

# Standardize the features
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data[features])

# ======= STEP 3: APPLY K-MEANS CLUSTERING =======
k = 4  # Adjust as needed
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

# Debugging: Check if 'Cluster' column exists
if 'Cluster' not in customer_data.columns:
    print("Error: 'Cluster' column missing. Re-running clustering...")
    customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

print("\nCluster labels assigned successfully.")
print("Unique clusters found:", customer_data['Cluster'].unique())

# ======= STEP 4: ANALYZE CLUSTERS =======
# View first few rows with assigned clusters
print("\nDataset with Cluster Labels:")
print(customer_data.head())

# Cluster-wise summary statistics
cluster_summary = customer_data.groupby('Cluster').mean()
print("\nCluster Summary:")
print(cluster_summary)

# ======= STEP 5: ANALYZE HIGH-VALUE CUSTOMERS =======
# Identify the highest-spending cluster
high_spending_cluster = cluster_summary['TotalSpent'].idxmax()
high_value_customers = customer_data[customer_data['Cluster'] == high_spending_cluster]

print(f"\nHigh-Value Customer Cluster: {high_spending_cluster}")
print(high_value_customers)

# ======= STEP 6: VISUALIZE CLUSTERS =======
# Boxplot to analyze spending per cluster
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='TotalSpent', data=customer_data)
plt.title("Spending Distribution by Cluster")
plt.show()

# Scatter plot for cluster visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=customer_data_scaled[:, 0], y=customer_data_scaled[:, 1], hue=customer_data['Cluster'], palette='viridis')
plt.xlabel("Standardized Total Spent")
plt.ylabel("Standardized Total Quantity")
plt.title("Customer Clusters")
plt.legend(title="Cluster")
plt.show()

# Save the updated dataset with clusters
customer_data.to_csv("clustered_customer_data.csv", index=False)
print("\nClustered data saved as 'clustered_customer_data.csv'")

