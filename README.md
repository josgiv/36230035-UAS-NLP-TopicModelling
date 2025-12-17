# 🎓 UAS Natural Language Processing: Topic Modeling

**Analisis Topic Modeling pada Dataset Blog Authorship Corpus**

---

## 👤 Identitas Mahasiswa
| Informasi | Detail |
|-----------|--------|
| **Nama** | Josia Given Santoso |
| **NIM** | 36230035 |
| **Mata Kuliah** | DSF06 – Natural Language Processing |
| **Kelas** | 5PDS3 |
| **Dosen** | [Nama Dosen Pengampu] |
| **Tanggal** | Desember 2025 |

### 💻 Spesifikasi Perangkat Keras
Proyek ini dikembangkan dan dioptimalkan khusus untuk berjalan pada spesifikasi berikut:
*   **Processor:** Intel Core i5 12450H (8 Core, 12 Thread)
*   **RAM:** 32GB

---

## 📋 Deskripsi Proyek
Proyek ini bertujuan untuk melakukan **Topic Modeling** (pemodelan topik) pada **Blog Authorship Corpus**, sebuah dataset besar yang berisi sekitar **681.288 dokumen** postingan blog. Menggunakan teknik *Unsupervised Learning*, proyek ini mengekstraksi tema atau topik tersembunyi dari teks blog yang tidak berlabel.

Notebook ini (`36230035_NLP_UAS_Optimized.ipynb`) telah dirancang dengan **Smart Engineering** untuk memastikan efisiensi memori dan kecepatan eksekusi yang optimal pada perangkat laptop standar, tanpa mengorbankan akurasi analisis.

---

## 🛠️ Metodologi & Model
Kami membandingkan empat algoritma populer untuk Topic Modeling:

1.  **LSA (Latent Semantic Analysis)**
    *   Pendekatan reduksi dimensi menggunakan SVD (Singular Value Decomposition).
    *   Cepat dan efisien untuk dataset besar.
2.  **LDA (Latent Dirichlet Allocation)**
    *   Model generatif probabilistik yang populer.
    *   Dioptimalkan menggunakan metode **'Online Learning'** (Mini-Batch) untuk menghemat RAM.
3.  **NMF (Non-negative Matrix Factorization)**
    *   Faktorisasi matriks non-negatif.
    *   Sering memberikan topik yang lebih koheren pada teks pendek/sedang.
4.  **BERTopic**
    *   Model state-of-the-art berbasis Transformer (BERT) dan c-TF-IDF.
    *   Dilatih menggunakan sampel representatif (statistically significant sample) karena beban komputasi yang berat.

---

## 🚀 Fitur & Optimasi Notebook
Proyek ini mengimplementasikan beberapa teknik optimasi tingkat lanjut:

*   **⚡ Dask Data Loading:** Menggunakan library `Dask` untuk memuat dataset CSV raksasa secara *lazy* dan mengonversinya ke format **Parquet** yang efisien, menghindari *MemoryError*.
*   **🚀 Multiprocessing Preprocessing:** Pipeline pembersihan teks (Cleaning, Lemmatization) dijalankan secara paralel menggunakan seluruh core CPU yang tersedia.
*   **🧠 Smart Hyperparameter Tuning:** Pencarian nilai optimal topik ($K$) menggunakan subset data (sampling) untuk mempercepat proses iterasi tanpa mengurangi validitas tren.
*   **📊 Visualisasi Lengkap:** Menyertakan Word Clouds, Bar Charts distribusi topik, KDE Plot distribusi usia, dan Interactive Topic Distance Map (pyLDAvis).

---

## 📊 Kesimpulan Utama
Berdasarkan evaluasi menggunakan **Coherence Score ($C_v$)**:

| Rank | Model | Coherence Score | Keterangan |
|------|-------|-----------------|------------|
| **1** | **NMF** | **0.4597** | **Best Performance.** Topik paling jelas dan terinterpretasi. |
| 2 | LSA | 0.4581 | Sangat cepat, hasil cukup baik. |
| 3 | LDA | 0.3942 | Performa terendah pada dataset sparse ini. |

**Topik yang Ditemukan (K=5):**
1.  **Daily Life/Work:** Time, day, work, going, home.
2.  **Politics/Government:** Bush, government, war, president, state.
3.  **General Chit-Chat:** People, know, think, going, like.
4.  **Expression/Personal:** Oh, lol, yeah, god, love.
5.  **Time/Events:** Night, day, time, last, week.

---

## ⚙️ Cara Menjalankan Proyek
Ikuti langkah-langkah berikut untuk menjalankan proyek ini di mesin lokal Anda:

### 1. Persiapan Environment
Pastikan Anda memiliki Python 3.9+ terinstal. Disarankan menggunakan virtual environment.

```bash
# Clone repository ini (jika ada)
# git clone ...

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 2. Instalasi Dependensi
Instal semua library yang dibutuhkan menggunakan `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Download Model SpaCy
Proyek ini menggunakan model bahasa Inggris dari SpaCy:

```bash
python -m spacy download en_core_web_sm
```

### 4. Jalankan Notebook
Buka Jupyter Notebook atau VS Code:

```bash
jupyter notebook Notebook/36230035_NLP_UAS_Optimized.ipynb
```

> **Catatan:** Saat pertama kali dijalankan, notebook akan otomatis mendownload dataset dari Kaggle (~300MB+). Pastikan koneksi internet stabil.

---

## 📂 Struktur Folder
```
.
├── Notebook/
│   ├── 36230035_NLP_UAS_Optimized.ipynb  # Notebook Utama
│   ├── preprocessing_helpers.py          # Script bantu untuk Multiprocessing
│   └── ...
├── datasets/                             # Folder penyimpanan data (otomatis dibuat)
├── README.md                             # Dokumentasi Proyek
└── requirements.txt                      # Daftar library python
```
