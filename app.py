# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ==============================
# KONFIGURASI
# ==============================

DATA_PATH = Path("df_convert.csv")
MODEL_PATH = Path("model_learner_classifier.pkl")
SCALER_PATH = Path("scaler.pkl")
LOGO_PATH = Path("logo.png")  # opsional

TARGET_COL = "learner_type"

# Fitur yang dipakai model (SAMAKAN dengan saat training)
FEATURE_COLS = [
    "avg_study_duration",
    "avg_submission_rating",
    "avg_exam_score",
    "total_submissions",
    "total_tracking_events",
    "total_completed_modules",   # sub-modul / tutorial yang selesai
    "days_since_last_active",
]

LABEL_DISPLAY = {
    0: "Consistent Learner",
    1: "Fast Learner",
    2: "Reflective Learner",
}

# ==============================
# FUNGSI UTIL
# ==============================

def to_float_safe(x, default=0.0):
    """Konversi string/koma ke float, kalau gagal -> default."""
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

    # MODEL
    if path_model.exists():
        try:
            if joblib is not None:
                model = joblib.load(path_model)
            else:
                with open(path_model, "rb") as f:
                    model = pickle.load(f)
        except Exception as e:
            st.error(f"Gagal load model: {e}")

    # SCALER
    if path_scaler.exists():
        try:
            if joblib is not None:
                scaler = joblib.load(path_scaler)
            else:
                with open(path_scaler, "rb") as f:
                    scaler = pickle.load(f)
        except Exception:
            scaler = None
    else:
        scaler = None

    return model, scaler


def predict_learner_type(model, scaler, feature_dict):
    """Scaling + prediksi label."""
    x = np.array([[feature_dict[c] for c in FEATURE_COLS]])
    if scaler is not None:
        x = scaler.transform(x)

    pred = model.predict(x)[0]
    return LABEL_DISPLAY.get(pred, str(pred))


def build_reason_sentence(learner_type: str, user_data: dict) -> str:
    """
    Penjelasan deskriptif + saran, terinspirasi dari snippet HTML yang kamu kirim,
    tapi disesuaikan dengan fitur yang tersedia sekarang.
    """
    avg_study_duration = user_data.get("avg_study_duration", 0.0)
    avg_submission_rating = user_data.get("avg_submission_rating", 0.0)
    avg_exam_score = user_data.get("avg_exam_score", 0.0)
    total_submissions = int(user_data.get("total_submissions", 0.0))
    total_tracking_events = int(user_data.get("total_tracking_events", 0.0))
    total_completed_modules = int(user_data.get("total_completed_modules", 0.0))
    days_since_last_active = int(user_data.get("days_since_last_active", 0.0))

    # FAST LEARNER
    if learner_type == "Fast Learner":
        text = (
            f"Kamu adalah **Fast Learner** karena kamu menunjukkan kecepatan yang tinggi "
            f"dalam menyelesaikan sub-modul/Tutorial (total sekitar **{total_completed_modules}** sub-modul selesai) "
            f"dengan rata-rata durasi belajar sekitar **{avg_study_duration:.2f} menit** per sesi. "
            f"Interaksimu dengan sistem juga sudah cukup banyak (**{total_tracking_events}** aktivitas) "
            f"dan kamu sudah mengumpulkan **{total_submissions}** tugas/kuis."
        )

        # kualitas nilai kurang
        if avg_submission_rating < 2.0 or avg_exam_score < 70.0:
            text += (
                f" Namun, perlu diperhatikan bahwa rata-rata nilai submissionmu "
                f"(**{avg_submission_rating:.2f}**) dan skor ujianmu (**{avg_exam_score:.2f}**) "
                "masih cenderung lebih rendah. Ini bisa jadi tanda bahwa kecepatanmu "
                "sedikit mengorbankan pemahaman mendalam.\n\n"
                "**Saran:** Coba luangkan lebih banyak waktu untuk membaca ulang materi "
                "sebelum pindah ke sub-modul berikutnya, dan manfaatkan fitur review atau latihan tambahan "
                "untuk memperkuat konsep yang dirasa masih lemah."
            )
        else:
            text += (
                " Kecepatan belajarmu diimbangi dengan hasil yang cukup baik, baik dari sisi nilai "
                "submission maupun ujian.\n\n"
                "**Saran:** Pertahankan ritme belajar ini dan coba tantang dirimu dengan modul atau proyek "
                "yang lebih kompleks, misalnya dengan studi kasus nyata atau tugas yang lebih menantang."
            )

        return text

    # CONSISTENT LEARNER
    if learner_type == "Consistent Learner":
        text = (
            "Kamu adalah **Consistent Learner** karena kamu belajar secara teratur dan seimbang. "
            f"Kamu sudah menyelesaikan sekitar **{total_completed_modules}** sub-modul, "
            f"mengumpulkan **{total_submissions}** submission, dan mencatat sekitar "
            f"**{total_tracking_events}** aktivitas pembelajaran. "
            f"Rata-rata nilai submissionmu (**{avg_submission_rating:.2f}**) dan skor ujianmu "
            f"(**{avg_exam_score:.2f}**) menunjukkan performa yang cukup solid. "
            f"Kamu juga relatif masih aktif, dengan jarak sekitar **{days_since_last_active} hari** "
            "sejak aktivitas terakhir."
        )

        if avg_exam_score >= 85.0 and avg_submission_rating >= 4.0:
            text += (
                " Performa belajar seperti ini sangat impresif.\n\n"
                "**Saran:** Kamu bisa mulai mencoba peran sebagai mentor kecil-kecilan, misalnya membantu teman "
                "yang kesulitan atau membuat rangkuman materi. Mengajarkan orang lain adalah salah satu cara "
                "terbaik untuk semakin memperdalam pemahamanmu."
            )
        else:
            text += (
                " Kamu sudah memiliki keseimbangan yang baik antara frekuensi belajar dan pemahaman materi.\n\n"
                "**Saran:** Terus pertahankan pola belajarmu. Jika ada topik yang terasa lebih sulit, "
                "jangan ragu untuk mengulang sub-modul terkait atau mencari sumber belajar tambahan "
                "seperti dokumentasi resmi, video, atau diskusi dengan komunitas."
            )

        return text

    # REFLECTIVE LEARNER
    if learner_type == "Reflective Learner":
        text = (
            "Kamu adalah **Reflective Learner** karena aktivitas belajarmu cenderung belum terlalu tinggi, "
            f"namun kamu kemungkinan besar lebih banyak merenungkan dan memilih materi dengan hati-hati. "
            f"Saat ini, kamu baru menyelesaikan sekitar **{total_completed_modules}** sub-modul, "
            f"dengan total **{total_submissions}** submission dan **{total_tracking_events}** aktivitas pembelajaran. "
            f"Interval sekitar **{days_since_last_active} hari** sejak aktivitas terakhir juga menunjukkan "
            "bahwa ritme belajarmu masih bisa ditingkatkan."
        )

        text += (
            "\n\nIni bisa berarti kamu masih berada di tahap awal eksplorasi atau membutuhkan sedikit dorongan "
            "untuk lebih aktif.\n\n"
            "**Saran:** Coba pilih satu topik atau modul yang paling menarik buatmu dan fokuskan usaha "
            "untuk menyelesaikannya sampai tuntas. Setelah itu, evaluasi apa yang sudah berhasil dan bagian mana "
            "yang masih membingungkan. Jika kamu merasa kesulitan, jangan ragu untuk bertanya ke mentor, "
            "teman sekelas, atau komunitas belajar agar tidak merasa belajar sendirian."
        )

        return text

    # fallback
    return (
        f"Kamu mendapat kategori **{learner_type}** berdasarkan kombinasi pola aktivitas "
        "seperti durasi belajar, jumlah interaksi dengan sistem, tugas yang dikumpulkan, "
        "sub-modul yang selesai, dan nilai ujian rata-rata."
    )

# ==============================
# SETUP APLIKASI
# ==============================

st.set_page_config(page_title="EduInsight", layout="wide")

df = load_data(DATA_PATH)
model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

candidate_name_cols = [
    "student_name", "display_name", "name", "full_name", "nama",
    "username", "user_name", "learner_name", "user_id", "id"
]
name_col = next((c for c in candidate_name_cols if c in df.columns), df.columns[0])

# ==============================
# SIDEBAR (LOGO + NAMA PROJECT)
# ==============================

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=120)
    st.markdown("---")
    page = st.radio("Menu", ["Dashboard", "Inference"])

# ==============================
# HALAMAN DASHBOARD
# ==============================

if page == "Dashboard":
    st.header("📊 Dashboard Per Siswa")

    mentee_list = sorted(df[name_col].astype(str).unique())
    selected = st.selectbox("Pilih siswa", mentee_list)

    row = df[df[name_col].astype(str) == selected].iloc[-1]

    st.subheader(f"Profil – {selected}")

    all_cols = list(df.columns)
    for i in range(0, len(all_cols), 3):
        cols = st.columns(3)
        for j, c in enumerate(all_cols[i:i + 3]):
            with cols[j]:
                val = to_float_safe(row[c], None)
                display_val = int(val) if isinstance(val, float) and val.is_integer() else row[c]
                st.metric(c, display_val)

# ==============================
# HALAMAN INFERENCE
# ==============================

else:
    st.header("🧠 Inference Model")

    if model is None:
        st.warning("Model belum siap digunakan. Pastikan file model dan scaler sudah benar.")
    else:
        with st.expander("Penjelasan fitur yang digunakan model"):
            st.markdown(
                """
**total_tracking_events**  
Jumlah total aktivitas interaksi pengguna dengan sistem (event tracking seperti membuka tutorial, next, dsb).

**total_completed_modules**  
Dihitung dari **jumlah sub-modul / tutorial yang selesai**, bukan dari modul utama.

**total_submissions**  
Jumlah total tugas/kuis yang dikumpulkan.

**avg_submission_rating**  
Rata-rata rating dari seluruh submission.

**avg_study_duration**  
Rata-rata durasi belajar per sesi.

**avg_exam_score**  
Rata-rata nilai ujian.

**days_since_last_active**  
Jumlah hari sejak terakhir pengguna beraktivitas.
                """
            )

        st.subheader("Input Nilai Fitur")

        feature_values = {}
        cols = st.columns(3)
        for i, feat in enumerate(FEATURE_COLS):
            with cols[i % 3]:
                feature_values[feat] = st.number_input(feat, value=0.0)

        if st.button("🚀 Jalankan Prediksi"):
            try:
                learner_type = predict_learner_type(model, scaler, feature_values)

                st.success(f"Hasil Prediksi: **{learner_type}**")

                st.subheader("Kenapa kamu mendapat kategori ini?")
                explanation = build_reason_sentence(learner_type, feature_values)
                st.write(explanation)

            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")
