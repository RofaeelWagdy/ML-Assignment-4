import numpy as np

class KMeans:
    def __init__(self, n_clusters, init="kmeans++", max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol

    def _initialize_centroids_random(self, X):
        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[idx]

    def _initialize_centroids_kmeanspp(self, X):
        n_samples = X.shape[0]
        centroids = []

        # First centroid
        centroids.append(X[np.random.randint(n_samples)])

        for _ in range(1, self.n_clusters):
            # Distance squared to nearest centroid
            dist_sq = np.min(
                np.linalg.norm(X[:, None] - np.array(centroids), axis=2) ** 2,
                axis=1
            )

            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()

            next_centroid = X[np.searchsorted(cumulative_probs, r)]
            centroids.append(next_centroid)

        return np.array(centroids)

    def fit(self, X):
        # Initialize centroids
        if self.init == "random":
            self.centroids_ = self._initialize_centroids_random(X)
        elif self.init == "kmeans++":
            self.centroids_ = self._initialize_centroids_kmeanspp(X)
        else:
            raise ValueError("init must be 'random' or 'kmeans++'")

        self.inertia_history_ = []

        for _ in range(self.max_iter):
            # Assign clusters
            distances = np.linalg.norm(
                X[:, None] - self.centroids_[None, :], axis=2
            )
            labels = np.argmin(distances, axis=1)

            # Compute inertia
            inertia = np.sum(
                (X - self.centroids_[labels]) ** 2
            )
            self.inertia_history_.append(inertia)

            # Update centroids
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k)
                else self.centroids_[k]
                for k in range(self.n_clusters)
            ])

            # Check convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids_)
            self.centroids_ = new_centroids

            if centroid_shift < self.tol:
                break

        self.labels_ = labels
