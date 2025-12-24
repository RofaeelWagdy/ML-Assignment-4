import numpy as np

# --- Activation Functions ---
def relu(x): 
    return np.maximum(0, x)
def relu_deriv(x): 
    return (x > 0).astype(float)

def sigmoid(x): 
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_deriv(x): 
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x): 
    return np.tanh(x)
def tanh_deriv(x): 
    return 1 - np.tanh(x)**2

class Autoencoder:
    def __init__(self, layers, activations, lr=0.001, l2_reg=0.01):
        self.layers = layers 
        self.activations = activations
        self.lr = lr
        self.l2_reg = l2_reg
        self.weights = []
        self.biases = []
        
        # Xavier Initialization
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _get_act(self, name):
        if name == 'relu': return relu, relu_deriv
        if name == 'sigmoid': return sigmoid, sigmoid_deriv
        if name == 'tanh': return tanh, tanh_deriv
        return lambda x: x, lambda x: 1

    def forward(self, X):
        self.a = [X]
        self.z = []
        for i in range(len(self.weights)):
            act_fn, _ = self._get_act(self.activations[i])
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)
            self.a.append(act_fn(z))
        return self.a[-1]

    def get_bottleneck(self, X):
        # Bottleneck is the middle layer
        mid = len(self.layers) // 2
        out = X
        for i in range(mid):
            act_fn, _ = self._get_act(self.activations[i])
            out = act_fn(np.dot(out, self.weights[i]) + self.biases[i])
        return out

    def backward(self, X):
        m = X.shape[0]
        y_pred = self.a[-1]
        delta = (y_pred - X)
        
        for i in reversed(range(len(self.weights))):
            _, act_deriv = self._get_act(self.activations[i])
            delta = delta * act_deriv(self.z[i])
            
            grad_w = (np.dot(self.a[i].T, delta) / m) + (self.l2_reg * self.weights[i])
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            self.weights[i] -= self.lr * grad_w
            self.biases[i] -= self.lr * grad_b
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)

    def train(self, X, epochs=100, batch_size=32):
        loss_history = []
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            X_sh = X[idx]
            for i in range(0, len(X), batch_size):
                batch = X_sh[i:i+batch_size]
                self.forward(batch)
                self.backward(batch)
            loss_history.append(np.mean((self.forward(X) - X)**2))
        return loss_history

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