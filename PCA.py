import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # Mean center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[sorted_idx]
        self.eigenvectors_ = eigenvectors[:, sorted_idx]

        # Select top k components
        self.components_ = self.eigenvectors_[:, :self.n_components]

        # Explained variance ratio
        total_variance = np.sum(self.eigenvalues_)
        self.explained_variance_ratio_ = (
            self.eigenvalues_[:self.n_components] / total_variance
        )

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def inverse_transform(self, Z):
        return np.dot(Z, self.components_.T) + self.mean_

    def reconstruction_error(self, X):
        Z = self.transform(X)
        X_reconstructed = self.inverse_transform(Z)
        return np.mean((X - X_reconstructed) ** 2)
