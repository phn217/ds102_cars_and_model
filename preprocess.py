#Tiền xử lý dữ liệu xe bonbanh: làm sạch -> nhãn -> chia train/test -> feature -> chuẩn hóa
import numpy as np
import pandas as pd

CURRENT_YEAR = 2026
NUM_COLS = ["car_age", "log_km", "engine_l", "seats"]  # các cột số sẽ được chuẩn hóa

def load_and_clean(path: str = "cars_raw.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    # bỏ cột không dùng: id/url là định danh, variant là text tự do, model quá nhiều giá trị
    df = df.drop(columns=["id", "url", "variant", "model"])
    # bỏ dòng thiếu ở các cột categorical (chỉ vài dòng)
    df = df.dropna(subset=["color", "fuel", "gearbox"])
    # xe nhiều chỗ (xe khách, bus) không phải xe con -> bỏ
    df = df[df["seats"] <= 9]
    # km: xe mới = 0; giá trị rác (vd 4294967295) -> NaN
    df.loc[df["condition"] == "mới", "km"] = 0
    df.loc[df["km"] > 1_000_000, "km"] = np.nan
    # engine_l: xe điện không có dung tích -> 0; giá trị phi lý -> NaN
    df.loc[df["fuel"] == "điện", "engine_l"] = 0
    df.loc[df["engine_l"] > 8, "engine_l"] = np.nan
    return df.reset_index(drop=True)

def stratified_split(y: np.ndarray, test_size: float, rng: np.random.Generator):
    # chia train/test giữ nguyên tỉ lệ lớp. trả về (idx_train, idx_test)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_test = int(round(len(idx) * test_size))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    return rng.permutation(np.concatenate(train_idx)), rng.permutation(np.concatenate(test_idx))

def encode(df: pd.DataFrame, top_colors, top_provinces):
    # biến đổi thành bảng feature (số)
    out = pd.DataFrame(index=df.index)
    out["car_age"] = CURRENT_YEAR - df["year"]
    out["log_km"] = np.log1p(df["km"])
    out["engine_l"] = df["engine_l"]
    out["seats"] = df["seats"]
    # km thiếu thật sự ở xe cũ -> thêm cờ
    out["km_missing"] = df["km"].isna().astype(float)
    out["is_new"] = (df["condition"] == "mới").astype(float)
    out["imported"] = (df["origin"] == "nhập khẩu").astype(float)
    out["automatic"] = (df["gearbox"] == "tự động").astype(float)

    color = df["color"].where(df["color"].isin(top_colors), "khác")
    province = df["province"].where(df["province"].isin(top_provinces), "khác")
    cats = pd.DataFrame({"brand": df["brand"], "fuel": df["fuel"],
                         "color": color, "province": province})
    dummies = pd.get_dummies(cats, dtype=float)
    return pd.concat([out, dummies], axis=1)

def prepare_data(path: str = "cars_raw.csv", test_size: float = 0.2, seed: int = 42,
                 verbose: bool = False, standardize: bool = True):
    """
    Nhãn: y = log1p(giá, triệu đồng) -> hồi quy dự đoán giá xe
    standardize: True  -> z-score các cột số (NUM_COLS)
                 False -> giữ dữ liệu gốc (chưa chuẩn hóa)
    Trả về X_train, X_test, y_train, y_test, feature_names
    Các thống kê (median km, mean/std) chỉ tính trên tập train
    """
    df = load_and_clean(path)
    rng = np.random.default_rng(seed)

    # chia train/test trước. Chỉ dùng 4 nhóm giá (quartile) để chia đều xe rẻ/đắt vào train và test,
    # không phải nhãn của mô hình
    strat = pd.qcut(df["price_million"], 4, labels=False).to_numpy()
    tr, te = stratified_split(strat, test_size, rng)
    df_tr, df_te = df.iloc[tr], df.iloc[te]

    # log1p để giảm lệch phải của giá
    y_tr = np.log1p(df_tr["price_million"].to_numpy())
    y_te = np.log1p(df_te["price_million"].to_numpy())
    if verbose:
        print("Hồi quy: y = log(1 + giá triệu đồng)")

    # price_million k đưa vào để tránh rò rỉ nhãn
    top_colors = df_tr["color"].value_counts().head(6).index
    top_provinces = df_tr["province"].value_counts().head(5).index
    X_tr = encode(df_tr, top_colors, top_provinces)
    X_te = encode(df_te, top_colors, top_provinces).reindex(columns=X_tr.columns, fill_value=0.0)

    # điền thiếu bằng median của train 
    med = X_tr[NUM_COLS].median()
    X_tr[NUM_COLS] = X_tr[NUM_COLS].fillna(med)
    X_te[NUM_COLS] = X_te[NUM_COLS].fillna(med)

    # chuẩn hóa z-score cho cột số
    if standardize:
        mean, std = X_tr[NUM_COLS].mean(), X_tr[NUM_COLS].std().replace(0, 1)
        X_tr[NUM_COLS] = (X_tr[NUM_COLS] - mean) / std
        X_te[NUM_COLS] = (X_te[NUM_COLS] - mean) / std

    return (X_tr.to_numpy(float), X_te.to_numpy(float),
            y_tr, y_te, list(X_tr.columns))

def zscore(X_train: np.ndarray, X_test: np.ndarray):
    """
    Chuẩn hóa toàn bộ cột về mean = 0, variance = 1 (asm2)
    mean/std chỉ lấy từ train rồi áp dụng cho test
    Cột hằng số (std = 0) giữ std = 1 để không chia cho 0
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std

def find_collinear(X: np.ndarray, tol: float = 1e-6):
    """
    Tìm các cột phụ thuộc tuyến tính (cộng tuyến) bằng Gram-Schmidt từng cột
    Trả về (kept_idx, dropped_idx).
    """
    N, D = X.shape
    Q = np.ones((N, 1)) / np.sqrt(N)  # cơ sở trực chuẩn, bắt đầu từ hệ số chặn
    kept, dropped = [], []
    for j in range(D):
        v = X[:, j].astype(float)
        norm = np.linalg.norm(v)
        r = v - Q @ (Q.T @ v)
        r = r - Q @ (Q.T @ r)  # chiếu lần 2 cho ổn định số
        r_norm = np.linalg.norm(r)
        if norm > 0 and r_norm > tol * norm:
            kept.append(j)
            Q = np.hstack([Q, (r / r_norm)[:, None]])
        else:
            dropped.append(j)
    return kept, dropped

def drop_collinear(X_train: np.ndarray, X_test: np.ndarray, names, tol: float = 1e-6):
    """Loại cột cộng tuyến (xác định trên train) khỏi cả train và test (asm3)"""
    kept, dropped = find_collinear(X_train, tol)
    kept_names = [names[j] for j in kept]
    dropped_names = [names[j] for j in dropped]
    return X_train[:, kept], X_test[:, kept], kept_names, dropped_names


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, names = prepare_data(standardize=False)
    print("Train:", X_tr.shape, "| Test:", X_te.shape)
    print(f"log giá train: min {y_tr.min():.2f}, mean {y_tr.mean():.2f}, max {y_tr.max():.2f}")
    print(names)