import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
data = pd.read_csv("  ", sep="\t", index_col=0)
data_transposed = data.transpose()
labels = data_transposed.index.str.split("_").str[-1]
numeric_labels = np.where(labels == "Healthy group", 0, 1)
mlp = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
mlp.fit(data_transposed, numeric_labels)
feature_importances = np.abs(mlp.coefs_[0]).sum(axis=1)
sorted_gene_indices = feature_importances.argsort()[::-1]
sorted_genes = data_transposed.columns[sorted_gene_indices]
selected_genes_nn = sorted_genes[:15].tolist()
plt.figure(figsize=(10, 7))
plt.barh(selected_genes_nn[::-1], feature_importances[sorted_gene_indices][:15][::-1], color="skyblue")
plt.xlabel("Feature Importance")
plt.ylabel("Genes")
plt.title("Top 15 Genes Based on Neural Network Feature Importance")
plt.tight_layout()
plt.savefig("  ")
plt.close()
with open("   ", 'w') as file:
    for gene in selected_genes_nn:
        file.write(f"{gene}\n")
