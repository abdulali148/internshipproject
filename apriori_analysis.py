import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the prepared dataset
df = pd.read_csv("apriori_ready_data.csv")

# Convert transactions into a list of lists format
transactions = df['Description'].apply(lambda x: x.strip("[]").replace("'", "").split(", ")).tolist()

# Convert transactions into one-hot encoded format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Save the results
frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)
rules.to_csv("association_rules.csv", index=False)

print("Apriori analysis completed!")
print(f"Frequent itemsets saved as 'frequent_itemsets.csv'.")
print(f"Association rules saved as 'association_rules.csv'.")
