import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # Mean
        self.mu_ = np.mean(X, axis=0)

        # Center data
        X_centered = X - self.mu_

        # Covariance matrix
        C = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store top-k
        self.eigenvalues_ = eigenvalues[:self.n_components]
        self.U_ = eigenvectors[:, :self.n_components]

        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues_ / total_variance

    def transform(self, X):
        X_centered = X - self.mu_
        Z = self.U_.T @ X_centered.T
        return Z.T

    def inverse_transform(self, Z):
        return (self.U_ @ Z.T).T + self.mu_

    def reconstruction_error(self, X,Z):
        X_hat = self.inverse_transform(Z)
        return np.mean((X - X_hat) ** 2)
