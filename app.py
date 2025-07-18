import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

# Daftar optimizer yang tersedia
optimizer_options = ['adam', 'sgd', 'rmsprop']

st.title("Aplikasi Klasifikasi Kue dengan Pilihan Optimizer")

# Pilihan optimizer
optimizer_choice = st.selectbox("Pilih optimizer saat inferensi", optimizer_options)

# Muat model sesuai optimizer yang dipilih (asumsikan kamu punya model berbeda untuk tiap optimizer)
@st.cache(allow_output_mutation=True)
def load_model(optimizer):
    # Contoh: muat model berbeda tergantung optimizer
    # Ganti path sesuai model kamu
    model_path = f'path_ke_model_{optimizer}.h5'
    model = tf.keras.models.load_model(model_path)
    return model

# Muat model sesuai pilihan
model = load_model(optimizer_choice)

st.write(f"Model yang digunakan: {optimizer_choice.upper()}")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar kue", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    # Preprocessing gambar sesuai model
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediksi
    pred = model.predict(img_array)
    pred_index = np.argmax(pred)
    kelas_kue = ['Kue 1', 'Kue 2', 'Kue 3', 'Kue 4', 'Kue 5', 'Kue 6', 'Kue 7', 'Kue 8']
    kelas_terpilih = kelas_kue[pred_index]
    confidence = pred[0][pred_index]
    
    st.write(f"Klasifikasi: **{kelas_terpilih}**")
    st.write(f"Kepercayaan: {confidence:.2f}")
