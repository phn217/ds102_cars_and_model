# asm3 - loại các cột phụ thuộc tuyến tính (cộng tuyến) rồi so sánh Linear Regression
import numpy as np
from preprocess import prepare_data, drop_collinear
from linear_regression import LinearRegression
from metrics import print_comparison

def gram_info(X: np.ndarray):
    # rank và condition number của X^T X (có cột bias)
    Xb = LinearRegression.add_bias(X)
    return np.linalg.matrix_rank(Xb), Xb.shape[1], np.linalg.cond(Xb.T @ Xb)

def run(solver, X_train, y_train, X_test, y_test):
    try:
        m = LinearRegression(solver=solver).fit(X_train, y_train)
    except np.linalg.LinAlgError as e:
        print(f"  ({solver}) lỗi: {e}")
        return None
    return {"train": m.evaluate(X_train, y_train), "test": m.evaluate(X_test, y_test)}

def main():
    X_train, X_test, y_train, y_test, names = prepare_data(
        "cars_raw.csv", standardize=False)
    P_train, P_test, kept, dropped = drop_collinear(X_train, X_test, names)

    print(f"Số cột: {X_train.shape[1]} -> {P_train.shape[1]} sau khi loại cột cộng tuyến")
    print("Các cột bị loại:", dropped)
    print("(mỗi nhóm one-hot có tổng = 1 = cột bias nên một cột trong nhóm là dư thừa)")

    for tag, X in {"Gốc": X_train, "Đã loại cột": P_train}.items():
        rank, D, cond = gram_info(X)
        print(f"{tag:<12}: rank(X + bias) = {rank}/{D}, cond(X^T X) = {cond:.3e}")

    results = {}
    for tag, (a, b) in {"gốc": (X_train, X_test), "đã loại cột": (P_train, P_test)}.items():
        for solver in ["inv", "lstsq"]:
            r = run(solver, a, y_train, b, y_test)
            if r is not None:
                results[f"{solver} - {tag}"] = r

    print()
    print_comparison(results)

if __name__ == "__main__":
    main()