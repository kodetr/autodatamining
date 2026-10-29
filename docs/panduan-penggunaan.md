# Panduan Lengkap AutoDataMining untuk Mahasiswa & Peneliti

Selamat datang di panduan resmi **AutoDataMining**! Pustaka ini dirancang agar proses penambangan data (*data mining*) dan pembelajaran mesin (*machine learning*) menjadi **intuitif, cepat, bebas kesalahan metodologis (*zero data leakage*), serta langsung siap untuk penulisan skripsi, tesis, dan publikasi jurnal ilmiah**.

---

## Daftar Isi
1. [Instalasi](#1-instalasi)
2. [Dataset Edukasi Bawaan (Langsung Praktik)](#2-dataset-edukasi-bawaan-langsung-praktik)
3. [Klasifikasi: Prediksi Kelulusan Mahasiswa](#3-klasifikasi-prediksi-kelulusan-mahasiswa)
4. [Regresi: Prediksi Harga Rumah](#4-regresi-prediksi-harga-rumah)
5. [Clustering: Segmentasi Profil Nasabah Bank](#5-clustering-segmentasi-profil-nasabah-bank)
6. [Aturan Asosiasi: Market Basket Analysis](#6-aturan-asosiasi-market-basket-analysis)
7. [Deteksi Anomali: Deteksi Transaksi Mencurigakan (Fraud)](#7-deteksi-anomali-deteksi-transaksi-mencurigakan-fraud)
8. [Automated Hyperparameter Tuning (`results.tune()`)](#8-automated-hyperparameter-tuning-resultstune)
9. [Fitur Publikasi Skripsi & LaTeX Otomatis](#9-fitur-publikasi-skripsi--latex-otomatis)
10. [Kamus Konsep Interaktif (`explain_concept`)](#10-kamus-konsep-interaktif-explain_concept)
11. [Modul Praktikum Google Colab / Jupyter Notebooks](#11-modul-praktikum-google-colab--jupyter-notebooks)

---

## 1. Instalasi

Untuk menggunakan library beserta fitur visualisasi grafik dan flowchart pipeline:

```bash
python -m pip install "autodatamining[visualization]"
```

Jika mengembangkan langsung dari source repository:
```bash
python -m pip install -e ".[visualization]"
```

---

## 2. Dataset Edukasi Bawaan (Langsung Praktik)

Salah satu kesulitan terbesar pemula dan mahasiswa adalah mencari file dataset CSV yang bersih dan cocok untuk algoritma tertentu. AutoDataMining menyelesaikan masalah ini dengan menyediakan fungsi `load_dataset()`:

```python
from autodatamining import load_dataset, list_datasets

print("Daftar dataset bawaan:", list_datasets())
# Output: ['kelulusan_mahasiswa', 'harga_rumah', 'segmentasi_nasabah', 'keranjang_belanja', 'deteksi_penipuan']
```

Semua dataset di atas bekerja **100% offline** tanpa perlu mengunduh file dari internet. Tinjau [Katalog Dataset Edukasi](dataset-edukasi.md) untuk detail setiap kolom.

---

## 3. Klasifikasi: Prediksi Kelulusan Mahasiswa

### Studi Kasus
Diberikan data akademik mahasiswa (`ipk`, `sks_semester_4`, `kehadiran_persen`, `ikut_organisasi`). Tentukan apakah mahasiswa akan lulus tepat waktu (`lulus_tepat_waktu`: `"Ya"` / `"Tidak"`).

### Kode Lengkap:
```python
from autodatamining import AutoDataMining, load_dataset

# 1. Muat dataset edukasi
data = load_dataset("kelulusan_mahasiswa")
print(data.head())

# 2. Periksa profil data (nilai kosong, kardinalitas, potensi masalah)
profil = AutoDataMining.profile(data, target="lulus_tepat_waktu")
print("Potensi masalah data:", profil["potential_issues"])

# 3. Jalankan AutoML Klasifikasi
results = AutoDataMining(
    task="classification",
    target="lulus_tepat_waktu",
    scoring="f1_weighted",
    cv=5,
    random_state=42,
).fit(data)

# 4. Lihat Peringkat Algoritma (Leaderboard)
print(results.leaderboard)

# 5. Baca Kesimpulan Naratif Otomatis (Bahasa Indonesia)
print(results.summary())

# 6. Salin Tabel Hasil Penelitian untuk Bab IV Skripsi
print(results.to_academic_table())

# 7. Tampilkan Visualisasi Flowchart Pipeline & Confusion Matrix
results.plot(kind="pipeline", save_path="pipeline_klasifikasi.png")
results.plot(kind="confusion_matrix", save_path="confusion_matrix.png")

# 8. Simpan Model Pipeline Lengkap (Preprocessing + Model)
results.save("model_kelulusan.joblib")
```

---

## 4. Regresi: Prediksi Harga Rumah

### Studi Kasus
Memperkirakan harga jual rumah (`harga_juta`) berdasarkan `luas_tanah`, `luas_bangunan`, `kamar_tidur`, dan `jarak_pusat_kota_km`.

### Kode Lengkap:
```python
from autodatamining import AutoDataMining, load_dataset

# 1. Muat data harga rumah
data = load_dataset("harga_rumah")

# 2. Latih berbagai estimator regresi (Linear, Ridge, Lasso, Random Forest, Gradient Boosting, dll.)
results = AutoDataMining(
    task="regression",
    target="harga_juta",
    scoring="r2",
    cv=5,
    random_state=42,
).fit(data)

# 3. Peringkat Model & Metrik Holdout
print(results.leaderboard)
print("Metrik Data Uji (R², MAE, RMSE):", results.metrics)

# 4. Kesimpulan Naratif
print(results.summary())

# 5. Diagram Residuals (Analisis Galat Prediksi)
results.plot(kind="residuals", save_path="residual_regresi.png")

# 6. Ekspor Kode Tabel LaTeX Siap Pakai di Overleaf
print(results.to_latex())
```

---

## 5. Clustering: Segmentasi Profil Nasabah Bank

### Studi Kasus
Mengelompokkan nasabah perbankan secara alami (*unsupervised*) berdasarkan `usia`, `pendapatan_tahunan_juta`, `skor_pengeluaran_1_100`, dan `saldo_rekening_juta`.

### Kode Lengkap:
```python
import pandas as pd
from autodatamining import AutoDataMining, load_dataset

# 1. Muat dataset nasabah
data = load_dataset("segmentasi_nasabah")

# 2. Jalankan algoritma clustering (K-Means, Agglomerative, DBSCAN)
results = AutoDataMining(
    task="clustering",
    models=["kmeans", "agglomerative", "dbscan"],
    random_state=42,
).fit(data)

# 3. Bandingkan kualitas klaster (Silhouette Score, Calinski-Harabasz, Davies-Bouldin)
print(results.leaderboard)

# 4. Analisis Profil Tiap Klaster (Insight Bisnis)
data_hasil = data.copy()
data_hasil["klaster"] = results.labels
print("\nProfil Rata-rata Tiap Kelompok Nasabah:")
print(data_hasil.groupby("klaster").mean().round(2))

# 5. Visualisasi Klaster dalam 2D (PCA Projection)
results.plot(kind="clusters", save_path="visualisasi_klaster.png")
```

---

## 6. Aturan Asosiasi: Market Basket Analysis

### Studi Kasus
Menemukan kombinasi belanjaan yang sering dibeli bersamaan dari struk belanja supermarket menggunakan algoritma **Apriori** dan **FP-Growth**.

### Kode Lengkap:
```python
from autodatamining import (
    AutoDataMining,
    load_dataset,
    TransactionEncoder,
    fpgrowth,
    association_rules,
)

# 1. Muat dataset transaksi belanja
transaksi = load_dataset("keranjang_belanja")

# 2. Cara Cepat: Menggunakan AutoDataMining
results = AutoDataMining(
    task="association",
    min_support=0.2,     # Muncul minimal di 20% total transaksi
    min_confidence=0.5,  # Tingkat kepastian aturan minimal 50%
).fit(transaksi)

print("Aturan Asosiasi Terkuat:")
print(results.rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

# 3. Visualisasi Scatter Plot Aturan Asosiasi (Support vs Confidence diberi warna Lift)
results.plot(kind="rules", save_path="aturan_asosiasi.png")
```

---

## 7. Deteksi Anomali: Deteksi Transaksi Mencurigakan (Fraud)

### Studi Kasus
Mengidentifikasi transaksi kartu/perbankan yang mencurigakan (*outlier / fraud*) dari atribut nominal, jam transaksi, jarak, dan frekuensi.

### Kode Lengkap:
```python
from autodatamining import AutoDataMining, load_dataset

# 1. Muat data transaksi
data = load_dataset("deteksi_penipuan")

# 2. Bandingkan metode deteksi anomali (Isolation Forest, One-Class SVM, LOF)
results = AutoDataMining(
    task="anomaly",
    models=["isolation_forest", "one_class_svm", "lof"],
    random_state=42,
).fit(data)

print(results.leaderboard)

# 3. Identifikasi data anomali (Label: -1 = Anomali, 1 = Normal)
data["status"] = ["Anomali (Fraud)" if l == -1 else "Normal" for l in results.labels]
print(data["status"].value_counts())

# Tampilkan transaksi anomali yang terjaring
anomali = data[data["status"] == "Anomali (Fraud)"]
print(anomali.head())

# 4. Kesimpulan Naratif
print(results.summary())
```

---

## 8. Automated Hyperparameter Tuning (`results.tune()`)

Setelah AutoDataMining menemukan model terbaik dari leaderboard, Anda dapat mengoptimalkan parameter model tersebut menggunakan `results.tune()`:

```python
# Lakukan tuning otomatis pada model terbaik
tuned = results.tune(method="random", n_iter=8, cv=3)

# 1. Cetak ringkasan perbandingan sebelum vs sesudah
print(tuned.summary())

# 2. Lihat tabel kenaikan skor & parameter optimal
print(tuned.comparison)
print("Parameter terbaik:", tuned.best_params)

# 3. Gunakan pipeline yang sudah dioptimalkan untuk memprediksi data baru
prediksi = tuned.predict(data_baru)
```

> [!TIP]
> **Bebas Data Leakage:** `results.tune()` hanya menjalankan validasi silang pada data pelatihan (*train set*) dan mengevaluasi hasil akhir pada data uji (*test holdout*). Ini menjamin objektivitas penelitian ilmiah Anda.

---

## 9. Fitur Publikasi Skripsi & LaTeX Otomatis

AutoDataMining dirancang khusus untuk memangkas waktu pembuatan laporan skripsi:

### A. Tabel Bab IV Skripsi Baku (`results.to_academic_table()`)
Menghasilkan tabel markdown rapi berisi Peringkat, Algoritma, Nilai Rata-rata Validasi Silang (Mean ± Std), Skor Data Uji, dan Waktu Eksekusi:
```python
print(results.to_academic_table())
```
*Hasil:*
| Peringkat | Algoritma | CV Data Latih | Skor Data Uji (Accuracy) | Waktu | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | Decision Tree | 0.8948 ± 0.030 | 0.9167 | 0.004s | ok |
| 2 | Gradient Boosting | 0.8945 ± 0.035 | 0.9167 | 0.037s | ok |
| ... | ... | ... | ... | ... | ... |

### B. Kode Tabel LaTeX Siap Salin (`results.to_latex()`)
Langsung tempel ke editor Overleaf atau naskah IEEE/SINTA:
```python
print(results.to_latex())
```

### C. Draf Pembahasan Skripsi Otomatis (`results.summary()`)
Menghasilkan analisis naratif siap kutip:
```text
"Berdasarkan pengujian komparatif beberapa algoritma klasifikasi, model decision_tree
menunjukkan performa terbaik dengan akurasi sebesar 91.7% pada data pengujian.
Evaluasi validasi silang mengonfirmasi stabilitas model tanpa indikasi overfitting yang signifikan,
sehingga model decision_tree direkomendasikan sebagai model akhir dalam penelitian ini."
```

---

## 10. Kamus Konsep Interaktif (`explain_concept`)

Jika Anda atau mahasiswa Anda bingung mengenai makna istilah machine learning, gunakan fungsi `explain_concept()`:

```python
from autodatamining import explain_concept, list_concepts

# Lihat daftar konsep yang tersedia
print(list_concepts())

# Minta penjelasan konsep
print(explain_concept("cross_validation"))
print(explain_concept("overfitting"))
print(explain_concept("lift"))
```

Setiap penjelasan mencakup:
1. **Definisi Formal:** Teori matematis yang lugas.
2. **Analogi Sehari-hari:** Contoh nyata yang mudah dicerna akal sehat.
3. **Tips Praktis untuk Skripsi/Riset:** Rekomendasi konkret dalam penulisan ilmiah.

---

## 11. Modul Praktikum Google Colab / Jupyter Notebooks

Untuk kegiatan praktikum di laboratorium kampus atau belajar mandiri, folder [`notebooks/`](../notebooks) telah menyediakan 5 modul interaktif siap pakai:
1. `01_Praktikum_Klasifikasi.ipynb`
2. `02_Praktikum_Regresi.ipynb`
3. `03_Praktikum_Clustering.ipynb`
4. `04_Praktikum_Aturan_Asosiasi.ipynb`
5. `05_Praktikum_Deteksi_Anomali.ipynb`

Notebook dapat langsung diunggah ke [Google Colab](https://colab.research.google.com/) atau dijalankan secara lokal dengan `jupyter notebook`.
