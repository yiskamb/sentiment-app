# Triase Sentimen Ulasan Aplikasi Mobile (CRISP-DM)

Klasifikasi sentimen biner (negatif/positif) ulasan aplikasi Gojek di Google Play Store.

## Versi Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yiskamb/sentiment-app/blob/main/proyek_triase_sentimen_gojek.ipynb)

## Versi Streamlit
Aplikasi: ISI_TAUTAN_STREAMLIT

## Isi repositori
- `proyek_triase_sentimen_gojek.ipynb`: analisis lengkap CRISP-DM
- `app.py`: prototipe Streamlit (melatih model dan melakukan prediksi)
- `ulasan_com.gojek.app.csv`: dataset ulasan
- `requirements.txt`: dependensi aplikasi Streamlit

## Menjalankan lokal
pip install -r requirements.txt
streamlit run app.py
