from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Cấu hình ứng dụng và đường dẫn dữ liệu
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Phân tích & Dự đoán giá đất",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MASTER_FILE = DATA_DIR / "Nhom2_GiaDat.xlsx"
TRAIN_FILE = DATA_DIR / "Nhom2_GiaDat_train.xlsx"
TEST_FILE = DATA_DIR / "Nhom2_GiaDat_test.xlsx"

TARGET = "Price"  # Giá trong dữ liệu gốc có đơn vị triệu đồng
NUMERIC_FEATURES = ["Area", "Width", "Length", "Alley Width", "Latitude", "Longitude"]
CATEGORICAL_FEATURES = ["Locality", "Position", "Direction", "Road Type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# -----------------------------------------------------------------------------
# Giao diện
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .main .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1450px;}
      .hero {
        padding: 1.35rem 1.5rem; border-radius: 16px;
        background: linear-gradient(120deg, #12304a 0%, #1b5b71 55%, #318477 100%);
        color: white; margin-bottom: 1rem;
      }
      .hero h1 {color: white; margin: 0 0 .35rem 0; font-size: 2rem;}
      .hero p {color: #e5f2f4; margin: 0; font-size: 1rem;}
      .note {padding: .75rem 1rem; border-left: 4px solid #2b8290;
             background: rgba(80,140,150,.10); border-radius: 6px;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22);
        padding: .8rem 1rem; border-radius: 12px;}
      .small-muted {font-size: .88rem; opacity: .78;}
    </style>
    <div class="hero">
      <h1>🏡 Phân tích & Dự đoán giá đất</h1>
      <p>Ứng dụng học máy với Linear Regression và K-Means Clustering trên bộ dữ liệu nhóm cung cấp.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Đọc và chuẩn hóa dữ liệu
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Đang đọc các file Excel...")
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc workbook tổng hợp và hai tập train/test được cung cấp."""
    missing = [p.name for p in (MASTER_FILE, TRAIN_FILE, TEST_FILE) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Không tìm thấy file dữ liệu: " + ", ".join(missing)
            + ". Hãy giữ nguyên thư mục data đi kèm ứng dụng."
        )

    all_land = pd.read_excel(MASTER_FILE, sheet_name="Du_lieu_dat")
    train = pd.read_excel(TRAIN_FILE, sheet_name=0)
    test = pd.read_excel(TEST_FILE, sheet_name=0)
    return all_land, train, test


def extract_locality(location: Any) -> str:
    """Trích tên phường/xã từ cột Location; không có thì trả về 'Khác'."""
    if pd.isna(location):
        return "Khác"
    text = str(location).strip()
    match = re.search(r"((?:Phường|Xã|Thị trấn|Thị xã)\s*[^,]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Khác"


def prepare_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa các cột cần thiết, giữ các trường còn lại để xem dữ liệu."""
    out = df.copy()
    for col in [TARGET, *NUMERIC_FEATURES]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    if "Location" in out.columns:
        out["Locality"] = out["Location"].map(extract_locality)
    else:
        out["Locality"] = "Khác"

    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = "Không rõ"
        out[col] = out[col].fillna("Không rõ").astype(str).replace({"nan": "Không rõ", "": "Không rõ"})

    return out


def get_training_bounds(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Ngưỡng lọc ngoại lệ học từ train để không dùng thông tin của test."""
    bounds: dict[str, tuple[float, float]] = {}
    for col in [TARGET, "Area", "Width", "Length", "Alley Width"]:
        values = pd.to_numeric(train[col], errors="coerce") if col in train else pd.Series(dtype=float)
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        values = values[values > 0]
        if len(values) == 0:
            bounds[col] = (0.0, float("inf"))
        elif col in (TARGET, "Area"):
            bounds[col] = (float(values.quantile(0.005)), float(values.quantile(0.995)))
        else:
            bounds[col] = (float(values.quantile(0.005)), float(values.quantile(0.995)))
    return bounds


def filter_for_regression(
    raw_df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    require_target: bool = True,
) -> pd.DataFrame:
    """Lọc giá/diện tích bất hợp lệ và thay kích thước bất thường bằng NaN."""
    df = prepare_raw_frame(raw_df)
    mask = df["Area"].notna() & (df["Area"] > 0)
    mask &= df["Area"].between(*bounds["Area"])
    if require_target:
        mask &= df[TARGET].notna() & (df[TARGET] > 0)
        mask &= df[TARGET].between(*bounds[TARGET])
    df = df.loc[mask].copy()

    for col in ["Width", "Length", "Alley Width"]:
        lo, hi = bounds[col]
        invalid = (df[col] <= 0) | (~df[col].between(lo, hi))
        df.loc[invalid, col] = np.nan

    # Tọa độ sai định dạng/ngoài khu vực Việt Nam được xem là thiếu.
    df.loc[~df["Latitude"].between(8, 24), "Latitude"] = np.nan
    df.loc[~df["Longitude"].between(102, 110), "Longitude"] = np.nan

    # Các biến phân loại rỗng được giữ ở nhóm "Không rõ".
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Không rõ").astype(str)
    return df


@st.cache_resource(show_spinner="Đang huấn luyện Linear Regression và đánh giá trên tập test...")
def train_linear_model(
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
) -> dict[str, Any]:
    train_clean_base = prepare_raw_frame(train_source)
    bounds = get_training_bounds(train_clean_base)
    train_clean = filter_for_regression(train_source, bounds, require_target=True)
    test_clean = filter_for_regression(test_source, bounds, require_target=True)

    if len(train_clean) < 30:
        raise ValueError("Số dòng train hợp lệ quá ít để huấn luyện mô hình.")
    if len(test_clean) < 5:
        raise ValueError("Số dòng test hợp lệ quá ít để đánh giá mô hình.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Không rõ")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=15)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    # Giá có phân phối lệch phải mạnh; hồi quy trên log(1 + Giá) giúp giảm ảnh hưởng
    # của các bất động sản có giá trị rất lớn. Kết quả được đổi về triệu đồng.
    log_linear = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", log_linear),
        ]
    )

    X_train = train_clean[ALL_FEATURES]
    y_train = train_clean[TARGET].astype(float)
    X_test = test_clean[ALL_FEATURES]
    y_test = test_clean[TARGET].astype(float)

    model.fit(X_train, y_train)
    predicted = np.maximum(model.predict(X_test), 0)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "R2_log": float(r2_score(np.log1p(y_test), np.log1p(predicted))),
        "Median_APE": float(np.median(np.abs(y_test.to_numpy() - predicted) / y_test.to_numpy())),
    }
    eval_df = pd.DataFrame(
        {
            "Giá thực tế (tỷ)": y_test.to_numpy() / 1000,
            "Giá dự đoán (tỷ)": predicted / 1000,
            "Sai số tuyệt đối (tỷ)": np.abs(y_test.to_numpy() - predicted) / 1000,
        }
    )
    return {
        "model": model,
        "bounds": bounds,
        "train_clean": train_clean,
        "test_clean": test_clean,
        "metrics": metrics,
        "eval_df": eval_df,
        "y_test": y_test.to_numpy(),
        "predicted": predicted,
    }


@st.cache_data(show_spinner="Đang chạy thuật toán K-Means...")
def run_kmeans(data_source: pd.DataFrame, n_clusters: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    df = prepare_raw_frame(data_source)
    df = df[(df[TARGET] > 0) & (df["Area"] > 0)].copy()
    df["PricePerSqm"] = df[TARGET] / df["Area"]

    # Giới hạn hai đầu phân phối để các dòng bất thường không kéo lệch tâm cụm.
    kmeans_limits = {
        col: (float(df[col].quantile(0.005)), float(df[col].quantile(0.995)))
        for col in [TARGET, "Area", "PricePerSqm"]
    }
    for col, (lo, hi) in kmeans_limits.items():
        df = df[df[col].between(lo, hi)]

    if len(df) < n_clusters * 3:
        raise ValueError("Không đủ dữ liệu hợp lệ cho số cụm đã chọn.")

    # Log-transform xử lý độ lệch; StandardScaler đưa ba đặc trưng về cùng thang đo.
    features = pd.DataFrame(
        {
            "log_gia": np.log1p(df[TARGET].astype(float)),
            "log_dien_tich": np.log1p(df["Area"].astype(float)),
            "log_gia_m2": np.log1p(df["PricePerSqm"].astype(float)),
        },
        index=df.index,
    )
    scaled = StandardScaler().fit_transform(features)
    cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = cluster_model.fit_predict(scaled)

    clustered = df.copy()
    clustered["Cụm"] = [f"Cụm {int(x) + 1}" for x in labels]
    clustered["Diện tích (m²)"] = clustered["Area"]
    clustered["Giá/m² (triệu)"] = clustered["PricePerSqm"]

    profile = (
        clustered.groupby("Cụm", as_index=False)
        .agg(
            **{
                "Số tin đăng": (TARGET, "size"),
                "Giá trung bình (tỷ)": (TARGET, lambda x: x.mean() / 1000),
                "Giá trung vị (tỷ)": (TARGET, lambda x: x.median() / 1000),
                "Diện tích TB (m²)": ("Area", "mean"),
                "Giá/m² TB (triệu)": ("PricePerSqm", "mean"),
            }
        )
        .sort_values("Cụm")
    )

    silhouette = float("nan")
    if len(np.unique(labels)) > 1:
        silhouette = float(
            silhouette_score(
                scaled,
                labels,
                sample_size=min(2500, len(labels)),
                random_state=42,
            )
        )
    return clustered, profile, silhouette


def format_billion(value_million: float) -> str:
    return f"{value_million / 1000:,.2f} tỷ đồng"


def display_metric_row(metrics: dict[str, float]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{metrics['MAE'] / 1000:,.2f} tỷ", help="Sai số tuyệt đối trung bình trên tập test, tính trên giá gốc.")
    c2.metric("RMSE", f"{metrics['RMSE'] / 1000:,.2f} tỷ", help="Căn bậc hai sai số bình phương trung bình trên tập test, tính trên giá gốc.")
    c3.metric("R² (log giá)", f"{metrics['R2_log']:.3f}", help="R² tính trên log(1 + Giá), phù hợp với mục tiêu mà mô hình được huấn luyện.")
    c4.metric("Sai số % trung vị", f"{metrics['Median_APE']:.1%}", help="Trung vị của sai số tuyệt đối chia cho giá thực tế.")


# -----------------------------------------------------------------------------
# Tải dữ liệu và huấn luyện mô hình
# -----------------------------------------------------------------------------
try:
    all_land, train_source, test_source = load_datasets()
except Exception as exc:
    st.error(f"Không thể tải dữ liệu: {exc}")
    st.stop()

try:
    lr_result = train_linear_model(train_source, test_source)
except Exception as exc:
    st.error(f"Không thể huấn luyện mô hình Linear Regression: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# Điều hướng
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Điều hướng")
    page = st.radio(
        "Chọn chức năng",
        ["Tổng quan", "Dự đoán giá đất", "Phân cụm K-Means", "Đánh giá Linear Regression", "Khám phá dữ liệu"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("**Nguồn dữ liệu**")
    st.caption(f"Dữ liệu đất: {len(all_land):,} dòng")
    st.caption(f"Train: {len(train_source):,} dòng · Test: {len(test_source):,} dòng")
    st.caption("Giá gốc: triệu đồng")
    st.divider()
    st.caption("Ứng dụng học thuật – kết quả dự đoán chỉ mang tính tham khảo, không thay thế thẩm định giá.")


# -----------------------------------------------------------------------------
# Trang: Tổng quan
# -----------------------------------------------------------------------------
if page == "Tổng quan":
    st.subheader("Tổng quan bộ dữ liệu")
    valid_overview = prepare_raw_frame(all_land)
    valid_overview = valid_overview[(valid_overview[TARGET] > 0) & (valid_overview["Area"] > 0)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng tin đăng đất", f"{len(all_land):,}")
    c2.metric("Giá trung vị", format_billion(valid_overview[TARGET].median()))
    c3.metric("Diện tích trung vị", f"{valid_overview['Area'].median():,.1f} m²")
    c4.metric("Số phường/xã", f"{valid_overview['Locality'].nunique():,}")

    st.markdown("### Hai thuật toán trong ứng dụng")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1. Linear Regression – Dự đoán")
        st.write(
            "Mô hình học mối quan hệ giữa giá đất và các đặc trưng như diện tích, chiều ngang, "
            "chiều dài, khu vực, vị trí, hướng và loại đường. Dữ liệu giá được biến đổi log khi huấn luyện "
            "rồi chuyển ngược về đơn vị triệu đồng khi dự đoán."
        )
        st.metric("Số mẫu train sau lọc", f"{len(lr_result['train_clean']):,}")
    with col2:
        st.markdown("#### 2. K-Means – Phân cụm")
        st.write(
            "Thuật toán gom các tin đăng có đặc điểm tương tự thành các cụm dựa trên giá, diện tích "
            "và giá trên mỗi mét vuông. Các biến được log-transform và chuẩn hóa trước khi phân cụm."
        )
        st.metric("Số mẫu dữ liệu đất", f"{len(valid_overview):,}")

    st.markdown("### Phân bố giá đất")
    price_view = valid_overview[TARGET] / 1000
    bins = [-np.inf, 1, 3, 5, 10, 20, 50, 100, np.inf]
    labels = ["<1 tỷ", "1–3 tỷ", "3–5 tỷ", "5–10 tỷ", "10–20 tỷ", "20–50 tỷ", "50–100 tỷ", ">100 tỷ"]
    price_groups = pd.cut(price_view, bins=bins, labels=labels, right=False)
    price_counts = price_groups.value_counts(sort=False).rename_axis("Khoảng giá").to_frame("Số tin đăng")
    st.bar_chart(price_counts)
    st.markdown(
        '<div class="note">Lưu ý: dữ liệu gốc ghi <b>Price</b> theo triệu đồng. Ứng dụng quy đổi sang tỷ đồng '
        'khi hiển thị để dễ đọc; mô hình vẫn sử dụng đơn vị gốc khi huấn luyện.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Trang: Dự đoán giá đất
# -----------------------------------------------------------------------------
elif page == "Dự đoán giá đất":
    st.subheader("Dự đoán giá một lô đất")
    st.write("Nhập các thông tin của lô đất. Mô hình sẽ trả về giá ước tính theo dữ liệu lịch sử.")

    train_clean = lr_result["train_clean"]
    bounds = lr_result["bounds"]
    locality_counts = train_clean["Locality"].value_counts()
    locality_options = locality_counts.index.tolist()
    if not locality_options:
        locality_options = ["Khác"]

    # Giá trị mặc định dựa trên trung vị của tập train đã làm sạch.
    default_area = float(train_clean["Area"].median())
    width_vals = train_clean["Width"].dropna()
    length_vals = train_clean["Length"].dropna()
    default_width = float(width_vals.median()) if len(width_vals) else 5.0
    default_length = float(length_vals.median()) if len(length_vals) else 20.0

    with st.form("prediction_form"):
        st.markdown("#### Thông tin kích thước")
        a1, a2, a3 = st.columns(3)
        area = a1.number_input("Diện tích (m²)", min_value=1.0, max_value=2_000_000.0, value=round(default_area, 1), step=1.0)
        width = a2.number_input("Chiều ngang (m)", min_value=0.0, max_value=500.0, value=round(default_width, 1), step=0.5)
        length = a3.number_input("Chiều dài (m)", min_value=0.0, max_value=5_000.0, value=round(default_length, 1), step=1.0)

        st.markdown("#### Thông tin khu vực và vị trí")
        b1, b2 = st.columns(2)
        locality = b1.selectbox("Phường/Xã", locality_options, index=0)
        position = b2.selectbox("Vị trí", ["Đường chính", "Trong hẻm", "Không rõ"])
        b3, b4 = st.columns(2)
        direction = b3.selectbox("Hướng", ["Không rõ", "Bắc", "Đông Bắc", "Đông", "Đông Nam", "Nam", "Tây Nam", "Tây", "Tây Bắc"])
        road_type = b4.selectbox("Loại đường", ["Không rõ", "Đường nhựa", "Đường bê tông", "Đường đất", "Đường đá"])
        alley_vals = train_clean["Alley Width"].dropna()
        latitude_vals = train_clean["Latitude"].dropna()
        longitude_vals = train_clean["Longitude"].dropna()
        alley_default = float(alley_vals.median()) if len(alley_vals) else 5.0
        lat_default = float(latitude_vals.median()) if len(latitude_vals) else 10.9
        lon_default = float(longitude_vals.median()) if len(longitude_vals) else 106.7
        st.markdown("#### Thông tin bổ sung (có thể giữ giá trị mặc định)")
        c5, c6, c7 = st.columns(3)
        alley_width = c5.number_input("Độ rộng hẻm (m)", min_value=0.0, max_value=100.0, value=round(alley_default, 1), step=0.5)
        latitude = c6.number_input("Vĩ độ", min_value=8.0, max_value=24.0, value=round(lat_default, 6), step=0.0001, format="%.6f")
        longitude = c7.number_input("Kinh độ", min_value=102.0, max_value=110.0, value=round(lon_default, 6), step=0.0001, format="%.6f")
        submitted = st.form_submit_button("🔎 Dự đoán giá đất", type="primary", use_container_width=True)

    if submitted:
        new_data = pd.DataFrame(
            [
                {
                    "Area": area,
                    "Width": width if width > 0 else np.nan,
                    "Length": length if length > 0 else np.nan,
                    "Alley Width": alley_width if alley_width > 0 else np.nan,
                    "Latitude": latitude,
                    "Longitude": longitude,
                    "Locality": locality,
                    "Position": position,
                    "Direction": direction,
                    "Road Type": road_type,
                }
            ]
        )
        prediction = float(max(lr_result["model"].predict(new_data[ALL_FEATURES])[0], 0))
        st.success("Đã hoàn tất dự đoán.")
        st.metric("Giá đất ước tính", format_billion(prediction))
        st.caption(f"Tương đương khoảng {prediction:,.0f} triệu đồng theo đơn vị của dữ liệu gốc.")

        area_lo, area_hi = bounds["Area"]
        if area < area_lo or area > area_hi:
            st.warning(
                f"Diện tích bạn nhập nằm ngoài khoảng dữ liệu train dùng để huấn luyện "
                f"({area_lo:,.1f}–{area_hi:,.1f} m²). Kết quả có thể kém tin cậy do mô hình phải ngoại suy."
            )
        st.info(
            "Đây là ước tính thống kê từ dữ liệu tin đăng, không phải giá giao dịch đã xác minh. "
            "Các yếu tố pháp lý, quy hoạch, tình trạng đất và thời điểm giao dịch có thể làm giá thực tế khác biệt."
        )

    with st.expander("Mô hình sử dụng những biến nào?"):
        st.write("**Biến số:** Diện tích, chiều ngang, chiều dài, độ rộng hẻm, vĩ độ và kinh độ.")
        st.write("**Biến phân loại:** Phường/Xã, vị trí (đường chính/trong hẻm), hướng và loại đường.")
        st.write("Các giá trị thiếu được điền bằng trung vị (biến số) hoặc nhóm 'Không rõ' (biến phân loại).")


# -----------------------------------------------------------------------------
# Trang: Phân cụm K-Means
# -----------------------------------------------------------------------------
elif page == "Phân cụm K-Means":
    st.subheader("Phân cụm tin đăng bằng K-Means")
    st.write("Thay đổi số cụm để quan sát các nhóm tin đăng có đặc điểm giá và diện tích tương đồng.")
    n_clusters = st.slider("Số cụm K", min_value=2, max_value=8, value=4, step=1)
    try:
        clustered, profile, silhouette = run_kmeans(all_land, n_clusters)
    except Exception as exc:
        st.error(f"Không thể chạy K-Means: {exc}")
        st.stop()

    k1, k2, k3 = st.columns(3)
    k1.metric("Số cụm", n_clusters)
    k2.metric("Số mẫu được phân cụm", f"{len(clustered):,}")
    k3.metric("Silhouette Score", f"{silhouette:.3f}" if np.isfinite(silhouette) else "Không tính được")

    st.markdown("### Quy mô và đặc trưng các cụm")
    profile_display = profile.copy()
    for col in ["Giá trung bình (tỷ)", "Giá trung vị (tỷ)", "Diện tích TB (m²)", "Giá/m² TB (triệu)"]:
        profile_display[col] = profile_display[col].map(lambda x: f"{x:,.2f}")
    st.dataframe(profile_display, use_container_width=True, hide_index=True)

    st.markdown("### Biểu đồ giá và diện tích theo cụm")
    chart_sample = clustered.sample(min(5000, len(clustered)), random_state=42)
    scatter_data = chart_sample[["Area", TARGET, "Cụm"]].rename(
        columns={"Area": "Diện tích (m²)", TARGET: "Giá (triệu đồng)"}
    )
    st.scatter_chart(
        scatter_data,
        x="Diện tích (m²)",
        y="Giá (triệu đồng)",
        color="Cụm",
        use_container_width=True,
    )

    st.markdown("### Diễn giải kết quả")
    st.write(
        "Mỗi dòng trong bảng là một cụm do K-Means tạo ra. Hãy so sánh giá trung vị, diện tích trung bình "
        "và giá/m² trung bình để mô tả nhóm đất. Số thứ tự cụm chỉ là nhãn kỹ thuật, không phải thứ hạng "
        "tốt/xấu hay đắt/rẻ cố định."
    )
    st.caption("Silhouette Score nằm trong khoảng -1 đến 1; giá trị cao hơn thường cho thấy các cụm tách biệt rõ hơn.")

    with st.expander("Cách K-Means được áp dụng"):
        st.write("1. Loại các bản ghi không có giá hoặc diện tích dương.")
        st.write("2. Tính thêm giá trên mỗi mét vuông (triệu đồng/m²).")
        st.write("3. Log-transform ba biến: giá, diện tích và giá/m²; sau đó chuẩn hóa bằng StandardScaler.")
        st.write("4. Chạy K-Means với random_state = 42 và n_init = 20.")


# -----------------------------------------------------------------------------
# Trang: Đánh giá Linear Regression
# -----------------------------------------------------------------------------
elif page == "Đánh giá Linear Regression":
    st.subheader("Đánh giá mô hình Linear Regression")
    st.write("Mô hình được huấn luyện trên file train và đánh giá trên file test được cung cấp riêng.")
    display_metric_row(lr_result["metrics"])
    st.caption(
        f"Số mẫu sau làm sạch: {len(lr_result['train_clean']):,} train và "
        f"{len(lr_result['test_clean']):,} test. Các ngưỡng loại ngoại lệ được tính từ train."
    )

    st.markdown("### Giá thực tế so với giá dự đoán")
    scatter_eval = lr_result["eval_df"]
    st.scatter_chart(
        scatter_eval,
        x="Giá thực tế (tỷ)",
        y="Giá dự đoán (tỷ)",
        use_container_width=True,
    )
    st.markdown("### Phân bố sai số trên tập test")
    err_bins = pd.cut(
        scatter_eval["Sai số tuyệt đối (tỷ)"],
        bins=[-0.001, 0.5, 1, 2, 5, 10, 25, np.inf],
        labels=["≤0,5", "0,5–1", "1–2", "2–5", "5–10", "10–25", ">25"],
    )
    err_counts = err_bins.value_counts(sort=False).rename_axis("Sai số (tỷ)").to_frame("Số mẫu")
    st.bar_chart(err_counts)
    st.caption("Do phân phối giá và sai số lệch phải, nên cần xem đồng thời MAE, RMSE và R² thay vì chỉ một chỉ số.")

    with st.expander("Cách đọc các chỉ số"):
        st.write("**MAE:** Sai lệch tuyệt đối trung bình; càng thấp càng tốt.")
        st.write("**RMSE:** Phạt mạnh hơn các sai số lớn; càng thấp càng tốt.")
        st.write("**R² (log giá):** Mức độ giải thích biến thiên của log(1 + Giá) trên tập test; được tính trên cùng thang biến đổi với mục tiêu huấn luyện.")
        st.write("**Sai số % trung vị:** Trung vị của |giá thực tế - giá dự đoán| / giá thực tế.")
        st.write("MAE và RMSE được tính trên giá gốc (triệu đồng) nên có thể lớn khi dữ liệu chứa các tin đăng giá trị rất cao.")
        st.write("Mô hình dùng log(1 + Price) khi học, sau đó chuyển kết quả về triệu đồng để tính sai số giá gốc.")


# -----------------------------------------------------------------------------
# Trang: Khám phá dữ liệu
# -----------------------------------------------------------------------------
elif page == "Khám phá dữ liệu":
    st.subheader("Khám phá dữ liệu đất")
    df = prepare_raw_frame(all_land)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Xem trước dữ liệu")
        preferred = [
            "Title", "Price", "Area", "Location", "Width", "Length", "Alley Width", "Position", "Direction", "Road Type"
        ]
        shown = [c for c in preferred if c in df.columns]
        preview = df[shown].head(100).copy()
        if "Price" in preview:
            preview["Price (tỷ) "] = preview["Price"] / 1000
            preview = preview.drop(columns=["Price"])
        st.dataframe(preview, use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Tỷ lệ thiếu của các trường mô hình")
        missing = df[ALL_FEATURES + [TARGET]].isna().mean().mul(100).sort_values(ascending=False)
        missing_df = missing.rename("Tỷ lệ thiếu (%)").to_frame()
        st.dataframe(missing_df.style.format({"Tỷ lệ thiếu (%)": "{:.1f}%"}), use_container_width=True)

    st.markdown("#### Thống kê mô tả")
    numeric_desc = df[[TARGET, *NUMERIC_FEATURES]].describe().T
    st.dataframe(numeric_desc.style.format("{:.2f}"), use_container_width=True)

    st.download_button(
        "⬇️ Tải dữ liệu đất đã đọc dưới dạng CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="du_lieu_dat.csv",
        mime="text/csv",
        use_container_width=False,
    )


# -----------------------------------------------------------------------------
# Chân trang
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "Ứng dụng minh họa hai thuật toán học máy trên bộ dữ liệu Nhom2_GiaDat. "
    "Kết quả phụ thuộc chất lượng, độ đầy đủ và tính đại diện của dữ liệu đầu vào."
)
