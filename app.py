import streamlit as st
import numpy as np
import pickle
import pandas as pd

# === Konfigurasi Page ===
st.set_page_config(
    page_title="Klasifikasi Obesitas",
    page_icon="🍔",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# === Menampilkan 3 Logo di Tengah secara Rapi ===
col_space1, col_logo1, col_logo2, col_logo3, col_space2 = st.columns([1, 2, 2, 2, 1])

with col_logo1:
    st.image("upg.png", width=100)

with col_logo2:
    st.image("fti.png", width=120)

with col_logo3:
    st.image("logo_informatika.png", width=300)


# === Styling CSS untuk estetika ===
st.markdown("""
    <style>
        .stButton > button {
            background-color: #007ACC;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton > button:hover {
            background-color: #005f99;
            color: #e6f7ff;
        }
        .st-success {
            background-color: #d4edda;
            padding: 10px;
            border-radius: 8px;
            color: black;
        }
        
        /* Tambahan: memperbesar font semua elemen teks kecuali heading */
        html, body, .main, .css-18e3th9, .css-1d391kg, .css-ffhzg2 {
            font-size: 18px !important;
        }

        /* Kecualikan heading agar tetap seperti biasa */
        h1, h2, h3, h4, h5, h6 {
            font-size: revert !important;
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="main">', unsafe_allow_html=True)


# === Judul dan Deskripsi Aplikasi ===
st.markdown("""
    <div style='background-color: #007ACC; padding: 10px; border-radius: 10px; text-align: center;'>
        <h1 style='color: white;'>Klasifikasi Resiko Obesitas Berdasarkan Gaya Hidup</h1>
    </div>
""", unsafe_allow_html=True)

# === Tabel Kategori BMI ===
st.markdown("""
<p style='text-align: justify; font-size: 18px;'>
<strong>Kategori BMI (Body Mass Index):</strong>
</p>

<table style='width:100%; font-size: 16px; border-collapse: collapse;'>
  <thead style='background-color:#f2f2f2;'>
    <tr>
      <th style='text-align:left; padding: 8px;'>Kategori</th>
      <th style='text-align:left; padding: 8px;'>Rentang BMI (kg/m²)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style='padding: 8px;'>💤 Berat Kurang</td><td style='padding: 8px;'>&lt; 18.5</td></tr>
    <tr><td style='padding: 8px;'>✅ Berat Normal</td><td style='padding: 8px;'>18.5 - 24.9</td></tr>
    <tr><td style='padding: 8px;'>⚠️ Kelebihan Berat Badan Level I</td><td style='padding: 8px;'>25.0 - 27.9</td></tr>
    <tr><td style='padding: 8px;'>⚠️⚠️ Kelebihan Berat Badan Level II</td><td style='padding: 8px;'>28.0 - 29.9</td></tr>
    <tr><td style='padding: 8px;'>⚠️ Obesitas Tipe I</td><td style='padding: 8px;'>30.0 - 34.9</td></tr>
    <tr><td style='padding: 8px;'>⚠️⚠️ Obesitas Tipe II</td><td style='padding: 8px;'>35.0 - 39.9</td></tr>
    <tr><td style='padding: 8px;'>🚨 Obesitas Tipe III</td><td style='padding: 8px;'>&ge; 40.0</td></tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align: align left; font-size: 18px;'>Masukkan data di bawah ini dan lihat hasil BMI, klasifikasi & akurasi dari 3 algoritma Machine Learning yang digunakan:</p>", unsafe_allow_html=True)

# === Load Model dan Scaler ===
try:
    model_dict = pickle.load(open('all_models_new_gen2.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    st.error(f"❌ Gagal memuat model atau scaler: {e}")
    st.stop()

# === Input Pengguna ===
st.header("📥 Input Data Anda")
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Usia Anda (tahun)")
    Gender = st.selectbox("Jenis Kelamin Anda", ["Perempuan", "Laki-laki"])
    Height = st.number_input("Tinggi Badan Anda (m)")
    Weight = st.number_input("Berat Badan Anda (kg)")
    family_history_with_overweight = st.selectbox("Apakah Ada Riwayat Obesitas di Keluarga?", ["Tidak", "Ya"])
    FAVC = st.selectbox("Apakah Anda Sering Makan Makanan Tinggi Kalori?", ["Tidak", "Ya"])
    FCVC = st.selectbox("Seberapa Sering Anda Konsumsi Sayur?", ["Tidak Pernah", "Jarang", "Sering"])
    NCP = float(st.selectbox("Berapa Kali Jumlah Makan Utama Anda /Hari?", ["1", "3", "4"]))

with col2:
    CAEC = st.selectbox("Apakah Ada Cemilan di Antara Waktu Makan?", ["Selalu", "Sering", "Kadang-kadang", "Tidak Pernah"])
    SMOKE = st.selectbox("Apakah Anda Merokok?", ["Tidak", "Ya"])
    CH2O = float(st.selectbox("Berapa Liter Air Anda Konsumsi per Hari?", ["1", "2", "3"]))
    SCC = st.selectbox("Apakah Anda Pantau Kalori Harian?", ["Tidak", "Ya"])
    FAF = float(st.selectbox("Berapa Jam Anda Aktivitas Fisik/Minggu?", ["0", "1", "2", "3"]))
    TUE = float(st.selectbox("Berapa Jam Anda Pakai Gadget/Hari?", ["1", "2", "3"]))
    CALC = st.selectbox("Apakah Anda Konsumsi Alkohol?", ["Tidak Pernah", "Kadang Kadang", "Sering", "Selalu"])
    MTRANS = st.selectbox("Apa Transportasi Harian Anda?", ["Mobil", "Sepeda", "Motor", "Kendaraan Umum", "Jalan Kaki"])

# === Mapping Nilai Input ===
gender_val = 0 if Gender == "Perempuan" else 1
family_val = 0 if family_history_with_overweight == "Tidak" else 1
favc_val = 0 if FAVC == "Ya" else 1
fcvc_map = {"Tidak Pernah": 1, "Jarang": 2, "Sering": 3}
FCVC = fcvc_map[FCVC]
caec_map = {"Selalu": 0, "Sering": 1, "Kadang-kadang": 2, "Tidak Pernah": 3}
caec_val = caec_map[CAEC]
smoke_val = 0 if SMOKE == "Ya" else 1
scc_val = 0 if SCC == "Ya" else 1
calc_map = {"Tidak Pernah": 0, "Kadang Kadang": 1, "Sering": 2, "Selalu": 3}
calc_val = calc_map[CALC]
mtrans_map = {"Mobil": 0, "Sepeda": 1, "Motor": 2, "Kendaraan Umum": 3, "Jalan Kaki": 4}
mtrans_val = mtrans_map[MTRANS]


# === Mapping kelas prediksi ===
kategori_map = {
    0: '💤 Berat Kurang',
    1: '✅ Berat Normal',
    2: '⚠️ Obesitas Tipe I',
    3: '⚠️⚠️ Obesitas Tipe II',
    4: '🚨 Obesitas Tipe III',
    5: '⚠️ Kelebihan Berat Badan Level I',
    6: '⚠️⚠️ Kelebihan Berat Badan Level II'
}

# === Prediksi ===
if st.button("🔍 Lihat Hasilnya"):
    error_messages = []

    # === Validasi Usia ===
    if Age < 17:
        error_messages.append("❗ Usia minimal **17 tahun**.")
    elif Age > 100:
        error_messages.append("❗ Usia maksimal **100 tahun**.")

    # === Validasi Tinggi Badan ===
    if Height < 1.0:
        error_messages.append("❗ Tinggi badan minimal **1.0 meter**.")
    elif Height > 2.0:
        error_messages.append("❗ Tinggi badan maksimal **2.0 meter**.")

    # === Validasi Berat Badan ===
    if Weight < 20:
        error_messages.append("❗ Berat badan minimal **20 kg**.")
    elif Weight > 200:
        error_messages.append("❗ Berat badan maksimal  **200 kg**.")

    # === Tampilkan Pesan Error jika Ada ===
    if error_messages:
        for msg in error_messages:
            st.error(msg)
    else:

         # === Hitung BMI dan tampilkan kategori ===
        bmi = Weight / (Height ** 2)

        if bmi < 18.5:
            bmi_kategori = '💤 Berat Kurang'
        elif 18.5 <= bmi <= 24.9:
            bmi_kategori = '✅ Berat Normal'
        elif 25.0 <= bmi <= 27.9:
            bmi_kategori = '⚠️ Kelebihan Berat Badan Level I'
        elif 28.0 <= bmi <= 29.9:
            bmi_kategori = '⚠️⚠️ Kelebihan Berat Badan Level II'
        elif 30.0 <= bmi <= 34.9:
            bmi_kategori = '⚠️ Obesitas Tipe I'
        elif 35.0 <= bmi <= 39.9:
            bmi_kategori = '⚠️⚠️ Obesitas Tipe II'
        else:
            bmi_kategori = '🚨 Obesitas Tipe III'

        st.header("📊 Skor & Kategori BMI (Tinggi & Berat Badan)")
        st.success(f"**Skor BMI Anda: {bmi:.2f}**\n\nKategori: **{bmi_kategori}**")

        input_data = np.array([[Age, gender_val, Height, Weight, family_val,
                                favc_val, FCVC, NCP, caec_val, smoke_val,
                                CH2O, scc_val, FAF, TUE, calc_val, mtrans_val]])

        try:
            input_scaled = scaler.transform(input_data)

            st.header("📈 Hasil Dari 3 Algoritma Klasifikas Berdasarkan Gaya Hidup & Akurasi")
            for model_name in ['SVM', 'RandomForest', 'NaiveBayes']:
                try:
                    model = model_dict[model_name]["model"]
                    acc = model_dict[model_name]["accuracy"]
                    pred = model.predict(input_scaled)[0]
                    hasil = kategori_map.get(pred, "Kategori Tidak Diketahui")
                    st.success(f"**{model_name}** Hasilnya: **{hasil}**  \n🎯 Akurasi: **{acc:.2f}%**")
                except Exception as e:
                    st.error(f"❌ Model {model_name} gagal prediksi: {e}")

        except Exception as e:
            st.error(f"❌ Gagal melakukan normalisasi data: {e}")



