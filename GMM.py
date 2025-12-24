import numpy as np

class GMM:
    def __init__(self, n_components, cov_type='full', max_iter=100, tol=1e-4):
        self.k = n_components
        self.cov_type = cov_type
        self.max_iter = max_iter
        self.tol = tol

    def _gaussian_pdf(self, X, mean, cov):
        d = X.shape[1]
        cov += 1e-6 * np.eye(d)
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        diff = X - mean
        exponent = -0.5 * np.sum(np.dot(diff, inv) * diff, axis=1)
        return (1.0 / np.sqrt(((2 * np.pi)**d) * det)) * np.exp(exponent)

    def fit(self, X):
        n, d = X.shape
        self.weights = np.ones(self.k) / self.k
        self.means = X[np.random.choice(n, self.k, replace=False)]
        self.covs = [np.eye(d) for _ in range(self.k)]
        self.ll_history = []

        for _ in range(self.max_iter):
            # E-step
            probs = np.zeros((n, self.k))
            for j in range(self.k):
                probs[:, j] = self.weights[j] * self._gaussian_pdf(X, self.means[j], self.covs[j])
            
            sum_probs = np.sum(probs, axis=1, keepdims=True) + 1e-15
            resp = probs / sum_probs
            
            # M-step
            nk = np.sum(resp, axis=0)
            self.weights = nk / n
            
            # Update means
            for j in range(self.k):
                self.means[j] = np.sum(resp[:, [j]] * X, axis=0) / nk[j]

            # Update covariances
            if self.cov_type == 'tied':
                tied_cov = np.zeros((d, d))
                for j in range(self.k):
                    diff = X - self.means[j]
                    tied_cov += np.dot((resp[:, [j]] * diff).T, diff)
                tied_cov /= n
                self.covs = [tied_cov for _ in range(self.k)]
            
            else:
                for j in range(self.k):
                    diff = X - self.means[j]
                    if self.cov_type == 'full':
                        self.covs[j] = np.dot((resp[:, [j]] * diff).T, diff) / nk[j]
                    elif self.cov_type == 'diagonal':
                        self.covs[j] = np.diag(np.sum(resp[:, [j]] * (diff**2), axis=0) / nk[j])
                    elif self.cov_type == 'spherical':
                        avg_var = np.sum(resp[:, [j]] * (diff**2)) / (nk[j] * d)
                        self.covs[j] = np.eye(d) * avg_var

            curr_ll = np.sum(np.log(sum_probs))
            self.ll_history.append(curr_ll)
            if len(self.ll_history) > 1 and abs(curr_ll - self.ll_history[-2]) < self.tol:
                break
        
        # Calculate BIC/AIC (Parameters estimation depends on cov_type)
        if self.cov_type == 'full':
            cov_params = self.k * d * (d + 1) / 2
        elif self.cov_type == 'tied':
            cov_params = d * (d + 1) / 2
        elif self.cov_type == 'diagonal':
            cov_params = self.k * d
        elif self.cov_type == 'spherical':
            cov_params = self.k
            
        p = (self.k * d) + cov_params + (self.k - 1)
        self.bic = -2 * self.ll_history[-1] + p * np.log(n)
        self.aic = -2 * self.ll_history[-1] + 2 * p