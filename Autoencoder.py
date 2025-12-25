import numpy as np

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
        if name == 'relu': 
            return relu, relu_deriv
        if name == 'sigmoid': 
            return sigmoid, sigmoid_deriv
        if name == 'tanh': 
            return tanh, tanh_deriv
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
    
    def decode(self, Z):
        # Decoder starts from the middle layer
        mid = len(self.layers) // 2
        out = Z
        for i in range(mid, len(self.weights)):
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