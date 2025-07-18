import pandas as pd
import matplotlib.pyplot as plt

# Load frequent itemsets
df = pd.read_csv("frequent_itemsets.csv")

# Sort by support and take top 10 frequent itemsets
df = df.sort_values(by="support", ascending=False).head(10)

# Convert itemsets from string to a readable format
df["itemsets"] = df["itemsets"].apply(lambda x: x.replace("frozenset({", "").replace("})", "").replace("'", ""))

# Plot bar chart
plt.figure(figsize=(10, 5))
plt.barh(df["itemsets"], df["support"], color="skyblue")
plt.xlabel("Support")
plt.ylabel("Itemsets")
plt.title("Top 10 Frequent Itemsets")
plt.gca().invert_yaxis()  # Invert y-axis to show the highest at the top

# Show the plot
plt.show()
