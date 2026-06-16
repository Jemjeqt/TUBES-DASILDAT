"""
app_streaming.py
Prediksi Rating IMDb untuk Konten Streaming.

Input: rotten_tomatoes_score, awards_won, genre, platform, rating, release_year
Output: IMDb rating prediction dengan confidence interval

Usage:
    streamlit run app_streaming.py
"""

import streamlit as st
import glob
import joblib

st.cache_resource.clear()

st.set_page_config(
    page_title="Prediksi Rating IMDb",
    layout="centered"
)

# =============================================================================
# MAPPING KATEGORI
# =============================================================================

GENRE_MAP = {
    0: "Action", 1: "Adventure", 2: "Animation", 3: "Anime",
    4: "Comedy", 5: "Crime", 6: "Documentary", 7: "Drama",
    8: "Fantasy", 9: "Game Show", 10: "Horror", 11: "Kids",
    12: "Musical", 13: "Mystery", 14: "Reality TV", 15: "Romance",
    16: "Sci-Fi", 17: "Sports", 18: "Talk Show", 19: "Thriller", 20: "War"
}

PLATFORM_MAP = {
    0: "Netflix", 1: "Amazon Prime Video", 2: "Disney+", 3: "Hulu",
    4: "HBO Max", 5: "Paramount+", 6: "Apple TV+", 7: "Peacock",
    8: "JioCinema", 9: "Crunchyroll"
}

RATING_MAP = {
    0: "Adult (R, NC-17, TV-MA)",
    1: "Family (PG, TV-PG)",
    2: "Kids (G, TV-G, TV-Y)",
    3: "Teen (PG-13, TV-14)"
}

# =============================================================================
# LOAD MODEL
# =============================================================================

def get_model_files():
    return glob.glob("model/model*.joblib")

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("Prediksi Rating")

model_files = get_model_files()
if len(model_files) == 0:
    st.sidebar.error("Model tidak ditemukan!")
    st.stop()

model_names = [f.replace('model/', '').replace('.joblib', '') for f in model_files]
selected_model_name = st.sidebar.selectbox("Pilih Model:", model_names)
selected_model_path = model_files[model_names.index(selected_model_name)]

try:
    model = load_model(selected_model_path)
    st.sidebar.success(f"Model: {selected_model_name}")
except Exception as e:
    st.sidebar.error(f"Error: {e}")
    st.stop()

# =============================================================================
# HELPER
# =============================================================================

def lookup(mapping, selected):
    if selected.startswith("—"):
        return 0
    for k, v in mapping.items():
        if v == selected:
            return k
    return 0

def get_kategori(rating):
    if rating >= 7.5:
        return "Sangat Baik", "green"
    elif rating >= 6.5:
        return "Cukup Baik", "blue"
    elif rating >= 5.5:
        return "Rata-rata", "orange"
    else:
        return "Kurang", "red"

# =============================================================================
# UI
# =============================================================================

st.title("Prediksi Rating IMDb")
st.markdown(
    "Aplikasi ini memprediksi **rating IMDb** suatu konten streaming (film/serial) "
    "berdasarkan informasi seperti skor Rotten Tomatoes, genre, platform, jumlah award dan tahun rilis."
)
st.markdown("Masukkan informasi konten untuk prediksi rating IMDb.")

# Input
st.markdown("**Input Konten**")
col1, col2 = st.columns(2)

with col1:
    rotten = st.slider("Rotten Tomatoes Score (0-100):", 0, 100, 70, 1)
    awards = st.slider("Jumlah Award:", 0, 50, 0, 1)
    release_year = st.slider("Tahun Rilis:", 1990, 2026, 2023)

with col2:
    platform_selected = st.selectbox("Platform:", ["— Pilih —"] + list(PLATFORM_MAP.values()), key="platform")
    rating_selected = st.selectbox("Rating (Usia):", ["— Pilih —"] + list(RATING_MAP.values()), key="rating")
    genre_selected = st.selectbox("Genre:", ["— Pilih —"] + list(GENRE_MAP.values()), key="genre")

# Encode
genre_encoded = lookup(GENRE_MAP, genre_selected)
platform_encoded = lookup(PLATFORM_MAP, platform_selected)
rating_encoded = lookup(RATING_MAP, rating_selected)

# Check
if genre_selected == "— Pilih —" or platform_selected == "— Pilih —":
    st.info("Pilih Genre dan Platform terlebih dahulu")
    st.stop()

# Prediksi
st.markdown("---")
if st.button("Prediksi", type="primary", use_container_width=True):
    input_row = [
        rotten,
        awards,
        genre_encoded,
        platform_encoded,
        rating_encoded,
        release_year
    ]

    try:
        prediction = model.predict([input_row])[0]
        prediction = max(0.0, min(10.0, prediction))

        # Confidence interval (±0.6)
        confidence = 0.6
        pred_low = max(0.0, prediction - confidence)
        pred_high = min(10.0, prediction + confidence)

        kategori, color = get_kategori(prediction)

        st.markdown("---")
        st.markdown("**Hasil Prediksi**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("IMDb Rating", f"{prediction:.1f}")
        with col2:
            st.metric("Confidence Interval", f"{pred_low:.1f} - {pred_high:.1f}")

        st.markdown(f"**Kategori:** {kategori}")
        st.markdown(f"Estimasi: IMDb rating antara **{pred_low:.1f}** sampai **{pred_high:.1f}**")

        if color == "green":
            st.success(f"Konten berkualitas tinggi ({kategori})")
        elif color == "blue":
            st.info(f"Konten cukup baik ({kategori})")
        elif color == "orange":
            st.warning(f"Konten rata-rata ({kategori})")
        else:
            st.error(f"Konten kurang populer ({kategori})")

    except Exception as e:
        st.error(f"Error: {e}")