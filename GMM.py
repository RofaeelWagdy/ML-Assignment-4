import numpy as np

class GMM:
    def __init__(self, n_components, cov_type='full', max_iter=100, tol=1e-4):
        self.n_clusters = n_components
        self.cov_type = cov_type
        self.max_iter = max_iter
        self.tolerance = tol

    def _gaussian_pdf(self, X, mean, cov):
        dimension_count = X.shape[1]
        cov += 1e-6 * np.eye(dimension_count)
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        diff = X - mean
        exponent = -0.5 * np.sum(np.dot(diff, inv) * diff, axis=1)
        return (1.0 / np.sqrt(((2 * np.pi)**dimension_count) * det)) * np.exp(exponent)

    def fit(self, X):
        sample_count, dimension_count = X.shape
        self.weights = np.ones(self.n_clusters) / self.n_clusters
        self.means = X[np.random.choice(sample_count, self.n_clusters, replace=False)]
        self.covs = [np.eye(dimension_count) for _ in range(self.n_clusters)]
        self.log_likelihood_history = []

        for _ in range(self.max_iter):
            # E-step
            probs = np.zeros((sample_count, self.n_clusters))
            for j in range(self.n_clusters):
                probs[:, j] = self.weights[j] * self._gaussian_pdf(X, self.means[j], self.covs[j])
            
            # normalization of probalities (do that the sum = 1)
            sum_probs = np.sum(probs, axis=1, keepdims=True) + 1e-15
            resp = probs / sum_probs
            

            # M-step
            cluster_sample_count = np.sum(resp, axis=0)
            self.weights = cluster_sample_count / sample_count
            
            # Update means
            for j in range(self.n_clusters):
                self.means[j] = np.sum(resp[:, [j]] * X, axis=0) / (cluster_sample_count[j] + 1e-9)

            # Update covariances
            if self.cov_type == 'tied':
                tied_cov = np.zeros((dimension_count, dimension_count))
                for j in range(self.n_clusters):
                    diff = X - self.means[j]
                    tied_cov += np.dot((resp[:, [j]] * diff).T, diff)
                tied_cov /= sample_count
                self.covs = [tied_cov for _ in range(self.n_clusters)]
            
            else:
                for j in range(self.n_clusters):
                    diff = X - self.means[j]
                    if self.cov_type == 'full':
                        self.covs[j] = np.dot((resp[:, [j]] * diff).T, diff) / (cluster_sample_count[j] + 1e-9)
                    elif self.cov_type == 'diag':
                        self.covs[j] = np.diag(np.sum(resp[:, [j]] * (diff**2), axis=0) / (cluster_sample_count[j] + 1e-9))
                    elif self.cov_type == 'spherical':
                        avg_var = np.sum(resp[:, [j]] * (diff**2)) / (cluster_sample_count[j] + 1e-9)
                        self.covs[j] = np.eye(dimension_count) * avg_var

            current_log_likelihood = np.sum(np.log(sum_probs))
            self.log_likelihood_history.append(current_log_likelihood)
            if len(self.log_likelihood_history) > 1 and abs(current_log_likelihood - self.log_likelihood_history[-2]) < self.tolerance:
                break
        
        # Calculate BIC/AIC (Parameters estimation depends on cov_type)
        if self.cov_type == 'full':
            cov_params = self.n_clusters * dimension_count * (dimension_count + 1) / 2
        elif self.cov_type == 'tied':
            cov_params = dimension_count * (dimension_count + 1) / 2
        elif self.cov_type == 'diag':
            cov_params = self.n_clusters * dimension_count
        elif self.cov_type == 'spherical':
            cov_params = self.n_clusters
            
        parameter_count = (self.n_clusters * dimension_count) + cov_params + (self.n_clusters - 1)
        self.bic = -2 * self.log_likelihood_history[-1] + parameter_count * np.log(sample_count)
        self.aic = -2 * self.log_likelihood_history[-1] + 2 * parameter_count
        return self.log_likelihood_history