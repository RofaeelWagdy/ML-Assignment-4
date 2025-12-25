from PCA import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data      # shape (569, 30)
y = data.target    # 0 = malignant, 1 = benign
feature_names = data.feature_names

# Run PCA
pca = PCA(n_components=10)
pca.fit(X)
X_pca = pca.transform(X)
print("Shape after PCA:",X_pca.shape)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative variance:", np.sum(pca.explained_variance_ratio_))
print("Reconstruction error:", pca.reconstruction_error(X))


plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.show()