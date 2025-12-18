# 🧠 NLP Topic Discovery Engine: Blog Authorship Corpus Analysis

**Advanced Topic Modeling using NMF, LDA, and LSA on 681k Blog Posts**

---

## 📋 Deskripsi Proyek
Proyek ini bertujuan untuk melakukan **Topic Modeling** (pemodelan topik) pada **Blog Authorship Corpus**, sebuah dataset besar yang berisi sekitar **681.288 dokumen** blog. Menggunakan teknik *Unsupervised Learning*, proyek ini mengekstraksi tema atau topik tersembunyi dari teks blog yang tidak berlabel.

Notebook utama (`36230035_NLP_UAS_Optimized.ipynb`) dirancang dengan **Smart Engineering** untuk memastikan efisiensi memori dan kecepatan eksekusi yang optimal pada perangkat laptop standar melalui:
*   **Lazy Loading** dengan Dask & Parquet.
*   **Multiprocessing** untuk Text Preprocessing.
*   **Mini-Batch Training** untuk model Scikit-Learn.

Selain Notebook, proyek ini menyediakan **Dashboard Interaktif** berbasis Streamlit untuk mencoba model secara langsung.

---

## 🛠️ Metodologi & Model
Kami membandingkan empat algoritma populer:

1.  **NMF (Non-negative Matrix Factorization)** - 🏆 **Best Model**
    *   Memberikan topik paling koheren ($C_v$ Score: 0.4597).
    *   Cocok untuk data sparse (teknik faktorisasi matriks).
2.  **LSA (Latent Semantic Analysis)**
    *   Metode reduksi dimensi klasik (SVD). Sangat cepat.
3.  **LDA (Latent Dirichlet Allocation)**
    *   Model probabilistik generatif. Dilatih dengan metode *Online Learning*.
4.  **BERTopic**
    *   Model berbasis Transformer (state-of-the-art) untuk perbandingan kontekstual.

---

## 📊 Hasil Analisis Utama (K=5 Topik)
Berdasarkan evaluasi **NMF**, topik dominan yang ditemukan adalah:

| No | Label Topik | Kata Kunci | Deskripsi |
|----|-------------|------------|-----------|
| 1 | **Daily Life** | time, day, work, going | Rutinitas sehari-hari, sekolah, pekerjaan. |
| 2 | **Politics** | bush, government, war, state | Diskusi politik, pemilu AS, isu global. |
| 3 | **Chit-Chat** | people, know, think, like | Obrolan santai, opini umum. |
| 4 | **Personal** | oh, lol, yeah, god, love | Ekspresi emosi, curhat, bahasa gaul (slang). |
| 5 | **Time/Event** | night, day, time, last, week | Narasi berbasis waktu atau kejadian spesifik. |

---

## 🚀 Panduan Instalasi & Penggunaan

Ikuti langkah-langkah berikut untuk menjalankan **Notebook** maupun **Dashboard Streamlit**.

### 1. Persiapan Environment
Pastikan Python 3.9+ sudah terinstal. Buat virtual environment agar rapi:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 2. Instalasi Dependensi
Jalankan perintah ini untuk menginstal semua library:

```bash
pip install -r requirements.txt
```

*File `requirements.txt` mencakup library untuk Data Science (`pandas`, `numpy`), NLP (`spacy`, `nltk`, `gensim`), Visualisasi (`plotly`, `seaborn`), dan Web App (`streamlit`).*

### 3. Download Model Bahasa
Download model SpaCy dan NLTK corpus yang dibutuhkan:

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords wordnet omw-1.4
```

### 4. Menjalankan Notebook (Training)
Untuk melatih model ulang atau melihat proses analisis:

```bash
jupyter notebook Notebook/36230035_NLP_UAS_Optimized.ipynb
```
> **PENTING:** Jalankan seluruh sel di notebook untuk menghasilkan file model (`.joblib`) di folder `models/` yang diperlukan oleh dashboard.

---

## 🌐 Menjalankan Streamlit Dashboard

Setelah model tersimpan (langkah 4 selesai), jalankan dashboard interaktif:

```bash
streamlit run app/streamlit.py
```

**Fitur Dashboard:**
*   **Real-time Inference:** Masukkan teks bebas dan lihat prediksi topiknya.
*   **Pipeline Visibility:** Lihat proses cleaning, tokenization, dan TF-IDF secara transparan.
*   **Radar Chart:** Visualisasi probabilitas kecocokan ke 5 topik.
*   **Reasoning Engine:** Menyorot kata kunci pada teks input yang memengaruhi keputusan model.

---

## 📂 Struktur Direktori Proyek

```
.
├── models/                           # [AUTO] Tempat penyimpanan model hasil training
│   ├── best_model_nmf.joblib
│   └── tfidf_vectorizer.joblib
├── Notebook/
│   ├── 36230035_NLP_UAS_Optimized.ipynb   # Main Notebook
│   └── preprocessing_helpers.py           # Helper script multiprocessing
├── app/
│   └── streamlit.py                       # Source code Dashboard
├── datasets/                              # [AUTO] Download dataset Kaggle
├── requirements.txt                       # Dependency list
└── README.md                              # Dokumentasi ini
```

---
**NLP Final Exam Project - 2025**
