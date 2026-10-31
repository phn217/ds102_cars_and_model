import numpy as np
from tqdm import tqdm
from metrics import evaluate as _evaluate

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 500):
        self.w = None         
        self.lr = learning_rate
        self.epochs = epochs
        self.losses = []      

    @staticmethod
    def add_bias(X: np.ndarray):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    @staticmethod
    def sigmoid(z: np.ndarray):
        z = np.clip(z, -500, 500) 
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X: np.ndarray):
        return self.sigmoid(self.add_bias(X) @ self.w)

    def loss_fn(self, y_hat: np.ndarray, y: np.ndarray):
        eps = 1e-12
        return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    def fit(self, X: np.ndarray, y: np.ndarray, log_every: int = 0):
        """X: (N, D), y: (N, ) nhãn 0/1"""
        Xb = self.add_bias(X)   #(N, D+1)
        N, D = Xb.shape
        self.w = np.zeros(D)    #(D+1, )
        self.losses = []

        bar = tqdm(range(self.epochs), desc="Training")
        for epoch in bar:
            y_hat = self.sigmoid(Xb @ self.w)           
            self.losses.append(self.loss_fn(y_hat, y))
            if log_every and (epoch % log_every == 0 or epoch == self.epochs - 1):
                tqdm.write(f"epoch {epoch:5d} | loss = {self.losses[-1]:.4f}")
            grad = Xb.T @ (y_hat - y) / N               
            self.w -= self.lr * grad
            bar.set_postfix(loss=f"{self.losses[-1]:.4f}")
        return self

    def predict(self, X: np.ndarray, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        return _evaluate(y, self.predict(X), n_classes=2)