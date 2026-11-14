# asm1 - Linear Regression: dự đoán giá xe bonbanh trên dữ liệu gốc
import numpy as np
from preprocess import prepare_data
from linear_regression import LinearRegression
from metrics import regression_report, print_regression_report

def main():
    # standardize=False: dữ liệu gốc, chưa chuẩn hóa
    X_train, X_test, y_train, y_test, names = prepare_data(
        "cars_raw.csv", verbose=True, standardize=False)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # baseline: luôn dự đoán trung bình log giá của tập train
    baseline = np.full_like(y_test, y_train.mean())
    print("\n== Baseline (dự đoán trung bình) - Test ==")
    print_regression_report(regression_report(y_test, baseline))

    # các cột one-hot cộng tuyến -> dùng lstsq
    model = LinearRegression(solver="lstsq")
    model.fit(X_train, y_train)
    print(f"\nLoss (MSE/2, thang log) sau khi fit: {model.losses[-1]:.4f}")

    print("\n== Train ==")
    print_regression_report(model.evaluate(X_train, y_train))
    print("\n== Test ==")
    print_regression_report(model.evaluate(X_test, y_test))

    print("\nHệ số (thang log giá):")
    for n, w in zip(names[:8], model.w[:8]):
        print(f"  {n:<12} {w:+.4f}")
    print(f"  {'bias':<12} {model.w[-1]:+.4f}")

if __name__ == "__main__":
    main()