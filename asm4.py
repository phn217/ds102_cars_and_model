# asm4 - so sánh Linear Regression viết bằng numpy với scikit-learn
import numpy as np
from sklearn.linear_model import LinearRegression as SkLinearRegression
from preprocess import prepare_data, zscore, drop_collinear
from linear_regression import LinearRegression
from metrics import regression_report, print_comparison

def run_manual(model: LinearRegression, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    res = {"train": model.evaluate(X_train, y_train), "test": model.evaluate(X_test, y_test)}
    return res, model.predict(X_test), model.w

def run_sklearn(X_train, y_train, X_test, y_test):
    model = SkLinearRegression().fit(X_train, y_train)
    res = {"train": regression_report(y_train, model.predict(X_train)),
           "test": regression_report(y_test, model.predict(X_test))}
    return res, model.predict(X_test), np.append(model.coef_, model.intercept_)

def main():
    X_train, X_test, y_train, y_test, names = prepare_data(
        "cars_raw.csv", standardize=False)
    P_train, P_test, _, _ = drop_collinear(X_train, X_test, names)
    Z_train, Z_test = zscore(P_train, P_test)

    settings = {
        "Gốc": (X_train, X_test, LinearRegression("lstsq")),
        "Loại cột cộng tuyến": (P_train, P_test, LinearRegression("lstsq")),
        "Chuẩn hóa (GD lr=0.1)": (Z_train, Z_test,
                                  LinearRegression("gd", learning_rate=0.1, epochs=3000)),
    }

    results = {}
    for tag, (a, b, model) in settings.items():
        r_np, pred_np, w_np = run_manual(model, a, y_train, b, y_test)
        r_sk, pred_sk, w_sk = run_sklearn(a, y_train, b, y_test)
        results[f"numpy - {tag}"] = r_np
        results[f"sklearn - {tag}"] = r_sk
        print(f"{tag:<22} max|dự đoán numpy - sklearn| (test, thang log) = "
              f"{np.abs(pred_np - pred_sk).max():.2e}")
        if tag != "Gốc":  # dữ liệu gốc cộng tuyến nên hệ số không duy nhất
            print(f"{'':<22} max|hệ số numpy - sklearn|                     = "
                  f"{np.abs(w_np - w_sk).max():.2e}")

    print()
    print_comparison(results)

if __name__ == "__main__":
    main()