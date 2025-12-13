# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ==============================
# KONFIGURASI
# ==============================

DATA_PATH = Path("df_dashboard.csv")
MODEL_PATH = Path("best_model.pkl")
SCALER_PATH = Path("scaler.pkl")
LOGO_PATH = Path("logo.png")  # opsional

TARGET_COL = "learner_type"

FEATURE_COLS = [
    "avg_study_duration",
    "avg_submission_rating",
    "avg_exam_score",
    "total_submissions",
    "total_tracking_events",
    "total_completed_modules",
    "days_since_last_active",
]

FEATURE_ALIASES = {
    "avg_study_duration": "Rata-rata belajar (menit)",
    "avg_submission_rating": "Rata-rata rating submission (bintang)",
    "avg_exam_score": "Rata-rata nilai kuis/ujian",
    "total_submissions": "Total submission (tugas)",
    "total_tracking_events": "Total membuka/mengakses materi",
    "total_completed_modules": "Total menyelesaikan materi",
    "days_since_last_active": "Terakhir aktif (hari)",
}

LABEL_DISPLAY = {
    0: "Consistent Learner",
    1: "Fast Learner",
    2: "Reflective Learner",
}

# ==============================
# PRIVASI DATA
# ==============================

# Kolom sensitif TIDAK DITAMPILKAN
PRIVATE_COLS = [
    "email", "email_address",
    "phone", "phone_number", "no_telp", "no_hp", "telephone", "telp"
]

# ==============================
# FUNGSI UTIL
# ==============================

def to_float_safe(x, default=0.0):
    if pd.isna(x):
        return default
    if isinstance(x, str):
        x = x.replace(",", ".")
    try:
        return float(x)
    except Exception:
        return default


@st.cache_data
def load_data(path: Path):
    return pd.read_csv(path)


@st.cache_resource
def load_model_and_scaler(path_model: Path, path_scaler: Path):
    model, scaler = None, None
    try:
        import joblib
    except ImportError:
        joblib = None

    if path_model.exists():
        try:
            model = joblib.load(path_model) if joblib else pickle.load(open(path_model, "rb"))
        except Exception as e:
            st.error(f"Gagal load model: {e}")

    if path_scaler.exists():
        try:
            scaler = joblib.load(path_scaler) if joblib else pickle.load(open(path_scaler, "rb"))
        except Exception:
            scaler = None

    return model, scaler


def predict_learner_type(model, scaler, feature_dict):
    x = np.array([[feature_dict[c] for c in FEATURE_COLS]])
    if scaler is not None:
        x = scaler.transform(x)
    pred = model.predict(x)[0]
    return LABEL_DISPLAY.get(pred, str(pred))


def build_reason_sentence(learner_type: str, user_data: dict) -> str:
    avg_study_duration = user_data.get("avg_study_duration", 0.0)
    avg_submission_rating = user_data.get("avg_submission_rating", 0.0)
    avg_exam_score = user_data.get("avg_exam_score", 0.0)
    total_tracking_events = int(user_data.get("total_tracking_events", 0.0))
    total_completed_modules = int(user_data.get("total_completed_modules", 0.0))
    days_since_last_active = int(user_data.get("days_since_last_active", 0.0))

    if learner_type == "Fast Learner":
        return (
            f"Kamu termasuk **Fast Learner** karena mampu menyelesaikan "
            f"**{total_completed_modules}** materi dengan cepat "
            f"(rata-rata belajar **{avg_study_duration:.1f} menit**).\n\n"
            f"Nilai ujian rata-rata **{avg_exam_score:.1f}** menunjukkan "
            "pemahaman yang tetap baik.\n\n"
            "**Saran:** Tetap luangkan waktu untuk review materi."
        )

    if learner_type == "Consistent Learner":
        return (
            "Kamu adalah **Consistent Learner** dengan pola belajar stabil.\n\n"
            f"Aktivitas belajar **{total_tracking_events}** kali "
            f"dan terakhir aktif **{days_since_last_active} hari** lalu.\n\n"
            "**Saran:** Pertahankan konsistensi ini."
        )

    if learner_type == "Reflective Learner":
        return (
            "Kamu termasuk **Reflective Learner** dengan ritme belajar lebih lambat.\n\n"
            f"Rata-rata belajar **{avg_study_duration:.1f} menit** "
            f"dan baru menyelesaikan **{total_completed_modules}** materi.\n\n"
            "**Saran:** Tingkatkan konsistensi belajar secara bertahap."
        )

    return f"Kamu dikategorikan sebagai **{learner_type}**."

# ==============================
# SETUP APLIKASI
# ==============================

st.set_page_config(page_title="EduInsight", layout="wide")

df = load_data(DATA_PATH)
model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

candidate_name_cols = [
    "student_name", "display_name", "name", "full_name",
    "nama", "username", "user_name", "learner_name", "id"
]
name_col = next((c for c in candidate_name_cols if c in df.columns), df.columns[0])

# ==============================
# SIDEBAR
# ==============================

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=120)
    st.markdown("---")
    page = st.radio("Menu", ["Dashboard", "Inference"])

# ==============================
# DASHBOARD
# ==============================

if page == "Dashboard":
    st.header("📊 Dashboard Per Siswa")

    mentee_list = sorted(df[name_col].astype(str).unique())
    selected = st.selectbox("Pilih siswa", mentee_list)

    row = df[df[name_col].astype(str) == selected].iloc[-1]

    st.subheader(f"Profil – {selected}")

    # ⛔ Filter kolom privat
    visible_cols = [
        c for c in df.columns
        if c.lower() not in PRIVATE_COLS
    ]

    for i in range(0, len(visible_cols), 3):
        cols = st.columns(3)
        for j, c in enumerate(visible_cols[i:i + 3]):
            with cols[j]:
                val = to_float_safe(row[c], None)
                display_val = int(val) if isinstance(val, float) and val.is_integer() else row[c]
                st.metric(c, display_val)

# ==============================
# INFERENCE
# =============================

else:
    st.header("🧠 Inference Model")

    if model is None:
        st.warning("Model belum siap digunakan.")
    else:
        st.subheader("Input Nilai Fitur")

        feature_values = {}
        cols = st.columns(3)
        for i, feat in enumerate(FEATURE_COLS):
            with cols[i % 3]:
                feature_values[feat] = st.number_input(
                    FEATURE_ALIASES.get(feat, feat),
                    value=0.0
                )

        if st.button("🚀 Jalankan Prediksi"):
            learner_type = predict_learner_type(model, scaler, feature_values)
            st.success(f"Hasil Prediksi: **{learner_type}**")
            st.write(build_reason_sentence(learner_type, feature_values))
