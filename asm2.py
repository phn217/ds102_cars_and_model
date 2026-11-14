# asm2 - so sánh Linear Regression trên dữ liệu gốc và dữ liệu đã chuẩn hóa (mean 0, var 1)
import numpy as np
import matplotlib.pyplot as plt
from preprocess import prepare_data, zscore
from linear_regression import LinearRegression
from metrics import print_comparison

def run(model: LinearRegression, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    return {"train": model.evaluate(X_train, y_train),
            "test": model.evaluate(X_test, y_test)}

def main():
    X_train, X_test, y_train, y_test, names = prepare_data(
        "cars_raw.csv", standardize=False)
    Z_train, Z_test = zscore(X_train, X_test)

    # kiểm tra chuẩn hóa (bỏ qua cột hằng số vì var = 0)
    var = Z_train.var(axis=0)
    var = var[var > 0]
    print(f"Sau chuẩn hóa (train): max|mean| = {np.abs(Z_train.mean(axis=0)).max():.2e}, "
          f"variance từ {var.min():.4f} đến {var.max():.4f}")

    EPOCHS = 3000
    results, models = {}, {}

    # 1) nghiệm đóng: chuẩn hóa không đổi dự đoán (chỉ đổi hệ số)
    for tag, (a, b) in {"gốc": (X_train, X_test), "chuẩn hóa": (Z_train, Z_test)}.items():
        m = LinearRegression(solver="lstsq")
        results[f"lstsq - {tag}"] = run(m, a, y_train, b, y_test)

    # 2) gradient descent: cùng lr, epochs cho dữ liệu gốc và chuẩn hóa
    # (lr = 0.01 gần mức lớn nhất mà dữ liệu gốc còn chưa phân kỳ)
    m_raw = LinearRegression("gd", learning_rate=0.01, epochs=EPOCHS)
    results["GD lr=0.01 - gốc"] = run(m_raw, X_train, y_train, X_test, y_test)
    m_std = LinearRegression("gd", learning_rate=0.01, epochs=EPOCHS)
    results["GD lr=0.01 - chuẩn hóa"] = run(m_std, Z_train, y_train, Z_test, y_test)

    # 3) dữ liệu chuẩn hóa cho phép dùng lr lớn hơn nhiều
    m_std_big = LinearRegression("gd", learning_rate=0.1, epochs=EPOCHS)
    results["GD lr=0.1 - chuẩn hóa"] = run(m_std_big, Z_train, y_train, Z_test, y_test)

    print()
    print_comparison(results)

    plt.figure(figsize=(6, 4))
    plt.plot(m_raw.losses, label="gốc, lr=0.01")
    plt.plot(m_std.losses, label="chuẩn hóa, lr=0.01")
    plt.plot(m_std_big.losses, label="chuẩn hóa, lr=0.1")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE/2, thang log giá)")
    plt.title("Linear Regression (GD) - training loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_linear_gd.png", dpi=120)
    plt.show()

if __name__ == "__main__":
    main()