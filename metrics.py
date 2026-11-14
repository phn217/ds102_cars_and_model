import numpy as np

# Nhãn của mô hình là y = log1p(giá, triệu đồng)
# RMSE và MAE tính trên thang giá gốc (triệu đồng), R2 tính trên thang log

def log_invert_transform(y: np.ndarray):
    return np.exp(y) - 1

def regression_report(y_log: np.ndarray, y_hat_log: np.ndarray):
    """y_log, y_hat_log: giá trị ở thang log1p"""
    with np.errstate(over="ignore", invalid="ignore"):
        y = log_invert_transform(y_log)
        y_hat = log_invert_transform(y_hat_log)
        err = y - y_hat
        rmse = float((err ** 2).mean() ** 0.5)
        mae = float(np.abs(err).mean())
    ss_res = ((y_log - y_hat_log) ** 2).sum()
    ss_tot = ((y_log - y_log.mean()) ** 2).sum()
    return {"rmse": rmse, "mae": mae,
            "rmse_log": float((((y_log - y_hat_log) ** 2).mean()) ** 0.5),
            "r2": float(1 - ss_res / ss_tot)}

def print_regression_report(res: dict):
    print(f"{'RMSE (triệu)':>14}: {res['rmse']:.2f}")
    print(f"{'MAE (triệu)':>14}: {res['mae']:.2f}")
    print(f"{'RMSE (log)':>14}: {res['rmse_log']:.4f}")
    print(f"{'R2 (log)':>14}: {res['r2']:.4f}")

def print_comparison(results: dict):
    """results: {tên thí nghiệm: {"train": report, "test": report}}"""
    print(f"{'Thí nghiệm':<34}{'RMSE train':>12}{'RMSE test':>12}{'MAE test':>11}{'R2 test':>10}")
    def fmt(v, w, d=2, big=1e7):
        # số quá lớn (nghiệm phân kỳ do cộng tuyến) -> ký hiệu khoa học cho khỏi dính cột
        return f"{v:>{w}.2e}" if abs(v) >= big else f"{v:>{w}.{d}f}"
    for name, r in results.items():
        print(f"{name:<34}{fmt(r['train']['rmse'], 12)}{fmt(r['test']['rmse'], 12)}"
              f"{fmt(r['test']['mae'], 11)}{fmt(r['test']['r2'], 10, 4, big=1e3)}")