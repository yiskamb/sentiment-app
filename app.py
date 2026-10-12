import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

st.set_page_config(page_title="Triase Sentimen Ulasan Gojek")

CSV_FILE = "ulasan_com.gojek.app.csv"

SLANG_DICTIONARY = {
    "g bsa": "tidak bisa", "ga bisa": "tidak bisa", "gak bisa": "tidak bisa", "gk bisa": "tidak bisa",
    "apk": "aplikasi", "app": "aplikasi", "bgt": "banget", "ga": "tidak", "gak": "tidak", "gk": "tidak",
    "dpt": "dapat", "bodo": "bodoh", "orderan": "pesanan", "driver": "pengemudi", "cs": "layanan pelanggan",
    "eror": "kesalahan", "lemot": "lambat", "lelet": "lambat", "bgus": "bagus", "mantul": "mantap",
    "tp": "tetapi", "klo": "kalau", "sy": "saya", "bsa": "bisa", "oke": "baik",
}


def clean_text(text):
    if not isinstance(text, str) or not text:
        return ""
    text = text.lower()
    text = re.sub(r"https?:\/\/\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    for slang, formal in SLANG_DICTIONARY.items():
        if " " in slang:
            text = re.sub(r"\b" + re.escape(slang) + r"\b", formal, text)

    tokens = text.split()
    tokens = [SLANG_DICTIONARY.get(t, t) for t in tokens]
    return " ".join(tokens).strip()


@st.cache_resource(show_spinner="Melatih model, tunggu sekitar satu menit...")
def train_models():
    # Data Understanding
    df_raw = pd.read_csv(CSV_FILE, usecols=["Ulasan", "Rating"])
    jumlah_mentah = len(df_raw)
    distribusi_rating = df_raw["Rating"].value_counts().sort_index(ascending=False)

    # Data Preparation: buang rating 3, label biner
    df = df_raw.dropna(subset=["Ulasan", "Rating"])
    df = df[df["Rating"] != 3].copy()
    df["sentiment"] = df["Rating"].apply(lambda r: 0 if r in [1, 2] else 1)

    # Urutan sama dengan notebook: bersihkan semua, buang < 2 kata, baru sampling
    df["text_clean"] = df["Ulasan"].apply(clean_text)
    df = df[df["text_clean"].str.split().str.len() >= 2].copy()

    df_neg = df[df["sentiment"] == 0].sample(n=5000, random_state=42)
    df_pos = df[df["sentiment"] == 1].sample(n=5000, random_state=42)
    df_clean = pd.concat([df_neg, df_pos]).sample(frac=1, random_state=42).reset_index(drop=True)

    X = df_clean["text_clean"].astype(str).tolist()
    y = df_clean["sentiment"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Modeling
    kandidat = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "LinearSVC": LinearSVC(random_state=42, dual="auto"),
    }
    hasil, pipes = [], {}
    for nama, clf in kandidat.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2)),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro")
        hasil.append({
            "Model": nama,
            "Accuracy": accuracy_score(y_test, pred),
            "Macro Precision": p,
            "Macro Recall": r,
            "Macro F1": f1,
        })
        pipes[nama] = pipe

    df_hasil = pd.DataFrame(hasil).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    terbaik = df_hasil.loc[0, "Model"]
    pred_terbaik = pipes[terbaik].predict(X_test)

    return {
        "pipeline": pipes[terbaik],
        "nama_terbaik": terbaik,
        "hasil": df_hasil,
        "laporan": classification_report(
            y_test, pred_terbaik, target_names=["Negatif", "Positif"]
        ),
        "cm": confusion_matrix(y_test, pred_terbaik),
        "jumlah_mentah": jumlah_mentah,
        "distribusi_rating": distribusi_rating,
        "jumlah_bersih": len(df_clean),
        "jumlah_latih": len(X_train),
        "jumlah_uji": len(X_test),
    }


art = train_models()
model = art["pipeline"]

st.title("Triase Sentimen Ulasan Aplikasi")
st.write(
    "Prototipe klasifikasi sentimen ulasan pengguna aplikasi mobile "
    "(Negatif = keluhan, Positif = apresiasi) dengan metodologi CRISP-DM."
)

tab_prediksi, tab_evaluasi = st.tabs(["Prediksi", "Hasil Evaluasi Model"])

with tab_prediksi:
    contoh = {
        "(tulis sendiri)": "",
        "Contoh positif": "Aplikasi mantap banget, sangat membantu saat cari ojek buru-buru!",
        "Contoh negatif": "Kecewa banget, saldo gopay kepotong tapi pesanan malah gagal terus!",
        "Contoh bug": "Tolong perbaiki bug saat pembayaran sering eror dan keluar sendiri.",
    }
    pilihan = st.selectbox("Pilih contoh ulasan atau tulis sendiri:", list(contoh.keys()))
    ulasan = st.text_area("Teks ulasan:", value=contoh[pilihan], height=120)

    if st.button("Prediksi", type="primary"):
        bersih = clean_text(ulasan)
        if len(bersih.split()) < 2:
            st.warning("Masukkan ulasan minimal dua kata.")
        else:
            pred = int(model.predict([bersih])[0])
            if pred == 1:
                st.success("Sentimen: POSITIF (apresiasi)")
            else:
                st.error("Sentimen: NEGATIF (keluhan / bug)")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([bersih])[0]
                st.write(f"Keyakinan model: **{proba[pred] * 100:.2f}%**")
                st.bar_chart({"Negatif": [proba[0]], "Positif": [proba[1]]})
            else:
                skor = float(np.ravel(model.decision_function([bersih]))[0])
                st.write(f"Skor keputusan model: **{skor:.3f}** (di atas 0 = positif)")

            with st.expander("Teks setelah pembersihan"):
                st.code(bersih)

with tab_evaluasi:
    st.subheader("Data")
    st.write(
        f"Data mentah: **{art['jumlah_mentah']:,}** ulasan. "
        f"Data bersih seimbang: **{art['jumlah_bersih']:,}** sampel "
        f"(latih {art['jumlah_latih']:,}, uji {art['jumlah_uji']:,})."
    )
    st.write("Distribusi rating data mentah:")
    st.bar_chart(art["distribusi_rating"])

    st.subheader("Perbandingan model")
    st.dataframe(art["hasil"].style.format({
        "Accuracy": "{:.4f}", "Macro Precision": "{:.4f}",
        "Macro Recall": "{:.4f}", "Macro F1": "{:.4f}",
    }))
    st.write(f"Model terbaik: **{art['nama_terbaik']}**")

    st.subheader("Classification report")
    st.code(art["laporan"])

    st.subheader("Confusion matrix")
    st.dataframe(pd.DataFrame(
        art["cm"],
        index=["Aktual Negatif", "Aktual Positif"],
        columns=["Prediksi Negatif", "Prediksi Positif"],
    ))
