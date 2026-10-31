import numpy as np
from tqdm import tqdm
from metrics import evaluate as _evaluate

class SoftmaxRegression:
    def __init__(self, n_classes: int, learning_rate: float = 0.1, epochs: int = 500):
        self.C = n_classes
        self.W = None         
        self.lr = learning_rate
        self.epochs = epochs
        self.losses = []

    @staticmethod
    def add_bias(X: np.ndarray):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    @staticmethod
    def softmax(Z: np.ndarray):
        Z = Z - Z.max(axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / expZ.sum(axis=1, keepdims=True)

    def _one_hot(self, y: np.ndarray):
        Y = np.zeros((len(y), self.C))
        Y[np.arange(len(y)), y] = 1
        return Y

    def predict_proba(self, X: np.ndarray):
        return self.softmax(self.add_bias(X) @ self.W)

    def loss_fn(self, Y_hat: np.ndarray, Y: np.ndarray):
        return -np.mean(np.sum(Y * np.log(Y_hat + 1e-12), axis=1))

    def fit(self, X: np.ndarray, y: np.ndarray, log_every: int = 0):
        """X: (N, D), y: (N, )"""
        Xb = self.add_bias(X)   # (N, D+1)
        N, D = Xb.shape
        Y = self._one_hot(y)    # (N, C)                          
        self.W = np.zeros((D, self.C))  # (D+1, C)
        self.losses = []

        bar = tqdm(range(self.epochs), desc="Training")
        for epoch in bar:
            Y_hat = self.softmax(Xb @ self.W)
            self.losses.append(self.loss_fn(Y_hat, Y))
            if log_every and (epoch % log_every == 0 or epoch == self.epochs - 1):
                tqdm.write(f"epoch {epoch:5d} | loss = {self.losses[-1]:.4f}")
            grad = Xb.T @ (Y_hat - Y) / N   #(D+1, C)              
            self.W -= self.lr * grad
            bar.set_postfix(loss=f"{self.losses[-1]:.4f}")
        return self

    def predict(self, X: np.ndarray):
        return self.predict_proba(X).argmax(axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray, class_names=None):
        return _evaluate(y, self.predict(X), n_classes=self.C, class_names=class_names)
