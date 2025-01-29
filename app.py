import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Muat model yang dilatih
MODEL_PATH = "random_forest_model.joblib"
try:
    model = load(MODEL_PATH)
    st.sidebar.success("Model successfully loaded!")
except FileNotFoundError:
    st.sidebar.error("Model file not found. Please make sure it's named 'random_forest_model.joblib'.")
    model = None

# Judul aplikasi
st.title("Aplikasi Prediksi Mental Health")

# Input fitur pengguna
st.header("Masukkan Informasi Anda")

age = st.number_input("Age", min_value=1, max_value=100, value=18)
gender = st.selectbox("Choose your gender", ["Male", "Female"])
anxiety = st.selectbox("Do you have Anxiety?", ["Yes", "No"])
depression = st.selectbox("Do you have Depression?", ["Yes", "No"])
panic_attack = st.selectbox("Do you have Panic attack?", ["Yes", "No"])
marital_status = st.selectbox("Marital status", ["Yes", "No"])

# Mapping input ke format model
input_data = {
    'Age': age,
    'Choose your gender_Male': 1 if gender == "Male" else 0,
    'Do you have Anxiety?_Yes': 1 if anxiety == "Yes" else 0,
    'Do you have Depression?_Yes': 1 if depression == "Yes" else 0,
    'Do you have Panic attack?_Yes': 1 if panic_attack == "Yes" else 0,
    'Marital status_Yes': 1 if marital_status == "Yes" else 0
}

# Konversi ke DataFrame
input_df = pd.DataFrame([input_data])

# Pastikan input sesuai dengan fitur model
if model:
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Tampilkan prediksi saat tombol diklik
if st.button("Predict"):
    if model is not None:
        try:
            if not input_df.empty:
                # Cek fitur model vs input
                st.subheader("Debugging: Cek Fitur Model vs. Input")
                st.write("Fitur Model:", list(model.feature_names_in_))
                st.write("Fitur Input:", list(input_df.columns))

                # Lakukan prediksi
                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)[:, 1]  # Probabilitas kelas 1

                # Debugging hasil prediksi
                st.subheader("Debugging: Hasil Prediksi dan Probabilitas")
                st.write(f"Prediksi Model: {prediction[0]}")
                st.write(f"Probabilitas Positif: {probability[0]:.4f}")

                # Tampilkan hasil
                if prediction[0] == 1:
                    st.success("Prediksi: **Positif** (Menunjukkan masalah kesehatan mental)")
                else:
                    st.warning("Prediksi: **Negatif** (Tidak ada masalah kesehatan mental yang terdeteksi)")

                # Cek apakah model terlalu bias
                if hasattr(model, "classes_"):
                    class_counts = np.bincount(model.classes_)
                    st.subheader("Debugging: Distribusi Label di Model")
                    st.write(f"Jumlah Data Kelas 0: {class_counts[0]}")
                    st.write(f"Jumlah Data Kelas 1: {class_counts[1]}")

            else:
                st.error("Input data tidak valid. Mohon masukkan data yang sesuai.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membuat prediksi: {e}")
    else:
        st.error("Model tidak dimuat, tidak dapat membuat prediksi.")
