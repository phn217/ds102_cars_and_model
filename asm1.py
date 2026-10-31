# asm1 - logistic regression: phân loại xe đắt(1) hay rẻ(0)
import numpy as np
import matplotlib.pyplot as plt
from preprocess import prepare_data
from logistic_regression import LogisticRegression
from metrics import print_report

def main():
    X_train, X_test, y_train, y_test, names = prepare_data("binary", "cars_raw.csv", verbose=True)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    model = LogisticRegression(learning_rate=0.5, epochs=3000)
    model.fit(X_train, y_train, log_every=500)
    print(f"\nLoss ban đầu: {model.losses[0]:.4f} -> loss cuối: {model.losses[-1]:.4f}")
    print("Loss giảm đơn điệu qua mọi epoch:", bool(np.all(np.diff(model.losses) <= 1e-9)))
    
    print("\n== Train ==")
    print_report(model.evaluate(X_train, y_train))
    print("\n== Test ==")
    print_report(model.evaluate(X_test, y_test))

    plt.figure(figsize=(6, 4))
    plt.plot(model.losses)
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("Logistic Regression - training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_logistic.png", dpi=120)
    plt.show()

if __name__ == "__main__":
    main()