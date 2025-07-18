import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load association rules
rules = pd.read_csv("association_rules.csv")

# Create a directed graph
G = nx.DiGraph()

# Add edges based on association rules
for _, row in rules.iterrows():
    antecedents = row["antecedents"].strip("{}").split(", ")
    consequents = row["consequents"].strip("{}").split(", ")
    confidence = round(row["confidence"], 2)

    for ant in antecedents:
        for cons in consequents:
            G.add_edge(ant, cons, weight=confidence)

# Draw the graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)  # Positioning
edges = G.edges(data=True)
edge_labels = {(u, v): d["weight"] for u, v, d in edges}

nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=2000, font_size=10, font_weight="bold")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Association Rules Network")
plt.show()
