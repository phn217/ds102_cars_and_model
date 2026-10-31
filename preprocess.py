#Tiền xử lý dữ liệu xe bonbanh: làm sạch -> nhãn -> chia train/test -> feature -> chuẩn hóa
import numpy as np
import pandas as pd

CURRENT_YEAR = 2026
NUM_COLS = ["car_age", "log_km", "engine_l", "seats"]  # các cột số sẽ được chuẩn hóa

CLASS_NAMES_MULTI = ["rẻ", "trung bình", "khá", "cao cấp"]

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

def prepare_data(task: str = "binary", path: str = "cars_raw.csv",
                 test_size: float = 0.2, seed: int = 42, verbose: bool = False):
    """
    task = "binary": nhãn 1 nếu giá > median (xe đắt), 0 nếu ngược lại.
    task = "multi" : 4 phân khúc giá theo quartile (0..3).

    Trả về X_train, X_test, y_train, y_test, feature_names.
    Các thống kê (median giá, quartile, median km, mean/std) chỉ tính trên tập train.
    """
    df = load_and_clean(path)
    rng = np.random.default_rng(seed)

    # chia train/test trước, stratify theo nhãn tạm (4 phân khúc theo toàn bộ giá)
    strat = pd.qcut(df["price_million"], 4, labels=False).to_numpy()
    tr, te = stratified_split(strat, test_size, rng)
    df_tr, df_te = df.iloc[tr], df.iloc[te]

    # nhãn, ngưỡng lấy từ train
    if task == "binary":
        thr = df_tr["price_million"].median()
        y_tr = (df_tr["price_million"] > thr).astype(int).to_numpy()
        y_te = (df_te["price_million"] > thr).astype(int).to_numpy()
        if verbose:
            print(f"Nhãn nhị phân: 1 = đắt nếu giá > {thr:.0f} triệu, 0 = rẻ")
    elif task == "multi":
        qs = df_tr["price_million"].quantile([0.25, 0.5, 0.75]).to_numpy()
        y_tr = np.digitize(df_tr["price_million"], qs)
        y_te = np.digitize(df_te["price_million"], qs)
        if verbose:
            print("Ngưỡng 4 phân khúc (triệu): {qs[0]:.0f} | {qs[1]:.0f} | {qs[2]:.0f}")
    """
    else:
        raise ValueError("task phải là 'binary' hoặc 'multi'")
    """

    # feature (price_million k đưa vào để tránh rò rỉ nhãn)
    top_colors = df_tr["color"].value_counts().head(6).index
    top_provinces = df_tr["province"].value_counts().head(5).index
    X_tr = encode(df_tr, top_colors, top_provinces)
    X_te = encode(df_te, top_colors, top_provinces).reindex(columns=X_tr.columns, fill_value=0.0)

    # điền thiếu bằng median của train (chỉ còn log_km, engine_l) 
    med = X_tr[NUM_COLS].median()
    X_tr[NUM_COLS] = X_tr[NUM_COLS].fillna(med)
    X_te[NUM_COLS] = X_te[NUM_COLS].fillna(med)

    # chuẩn hóa z-score cho cột số
    mean, std = X_tr[NUM_COLS].mean(), X_tr[NUM_COLS].std().replace(0, 1)
    X_tr[NUM_COLS] = (X_tr[NUM_COLS] - mean) / std
    X_te[NUM_COLS] = (X_te[NUM_COLS] - mean) / std

    return (X_tr.to_numpy(float), X_te.to_numpy(float),
            y_tr, y_te, list(X_tr.columns))

if __name__ == "__main__":
    for t in ["binary", "multi"]:
        X_tr, X_te, y_tr, y_te, names = prepare_data(t)
        print(t, X_tr.shape, X_te.shape, "labels train:", np.bincount(y_tr))
    print(names)
