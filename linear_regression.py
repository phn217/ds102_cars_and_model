import numpy as np
from tqdm import tqdm
from metrics import regression_report

class LinearRegression:
    """
    Linear Regression, 3 cách giải:
      solver="inv"  : phương trình chuẩn w = (X^T X)^-1 X^T y không ổn định/lỗi khi các cột cộng tuyến.
      solver="lstsq": np.linalg.lstsq (SVD) -> nghiệm chuẩn nhỏ nhất, chạy được cả khi cộng tuyến
      solver="gd"   : Gradient Descent
    """
    def __init__(self, solver: str = "lstsq", learning_rate: float = 0.01, epochs: int = 1000):
        if solver not in ("inv", "lstsq", "gd"):
            raise ValueError("solver phải là 'inv', 'lstsq' hoặc 'gd'")
        self.w = None
        self.solver = solver
        self.lr = learning_rate
        self.epochs = epochs
        self.losses = []

    @staticmethod
    def add_bias(X: np.ndarray):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def loss_fn(self, y_hat: np.ndarray, y: np.ndarray):
        return 0.5 * np.mean((y_hat - y) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray, log_every: int = 0):
        """X: (N, D), y: (N, ) -- y ở thang log1p"""
        Xb = self.add_bias(X)   # (N, D+1)
        N, D = Xb.shape
        self.losses = []

        if self.solver == "inv":
            cov = Xb.T @ Xb # (D+1, D+1)
            inv_cov = np.linalg.inv(cov)    # (D+1, D+1)
            self.w = inv_cov @ Xb.T @ y     # (D+1, )
            self.losses.append(self.loss_fn(Xb @ self.w, y))
        elif self.solver == "lstsq":
            self.w = np.linalg.lstsq(Xb, y, rcond=None)[0]  # (D+1, )
            self.losses.append(self.loss_fn(Xb @ self.w, y))
        else:
            self.w = np.zeros(D)    # (D+1, )
            bar = tqdm(range(self.epochs), desc="Training", disable=not log_every)
            for epoch in bar:
                y_hat = Xb @ self.w # (N, )
                self.losses.append(self.loss_fn(y_hat, y))
                if log_every and (epoch % log_every == 0 or epoch == self.epochs - 1):
                    tqdm.write(f"epoch {epoch:5d} | loss = {self.losses[-1]:.4f}")
                grad = Xb.T @ (y_hat - y) / N   # (D+1, )
                self.w -= self.lr * grad
                bar.set_postfix(loss=f"{self.losses[-1]:.4f}")
        return self

    def predict(self, X: np.ndarray):
        return self.add_bias(X) @ self.w

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        return regression_report(y, self.predict(X))
