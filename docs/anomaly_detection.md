# Panduan Deteksi Anomali (Anomaly Detection)

Modul deteksi anomali di **AutoDataMining** digunakan untuk mengidentifikasi observasi tidak biasa, pencilan (*outliers*), atau potensi penipuan (*fraud detection*) yang menyimpang dari pola umum data.

---

## 1. Mode Tanpa Label (Unsupervised) - Paling Sering Digunakan

Gunakan dataset bawaan `"deteksi_penipuan"` untuk mendeteksi transaksi kartu yang mencurigakan tanpa memerlukan kolom label kebenaran:

```python
import pandas as pd
from autodatamining import AutoDataMining, load_dataset

# 1. Muat dataset transaksi keuangan
data = load_dataset("deteksi_penipuan")
print(data.head())

# 2. Jalankan deteksi anomali
results = AutoDataMining(
    task="anomaly",
    models=["isolation_forest", "one_class_svm", "lof"],
    random_state=42,
).fit(data)

# 3. Leaderboard Deteksi Anomali
print(results.leaderboard)

# 4. Filter Transaksi Anomali (Label: -1 = Anomali, +1 = Inlier/Normal)
data_hasil = data.copy()
data_hasil["status"] = ["Anomali" if lbl == -1 else "Normal" for lbl in results.labels]

print("\nDistribusi Hasil Deteksi:")
print(data_hasil["status"].value_counts())

print("\n10 Contoh Transaksi Anomali:")
print(data_hasil[data_hasil["status"] == "Anomali"].head(10))

# 5. Kesimpulan Naratif Otomatis
print(results.summary())

# 6. Visualisasi Sebaran Anomali (PCA Projection)
results.plot(kind="anomalies", save_path="sebaran_anomali.png")
```

---

## 2. Mode Berlabel (Supervised Validation)

Jika Anda memiliki data uji dengan label kebenaran ground-truth (misalnya kolom `is_fraud` di mana `1` = anomali, `0` = normal), Anda dapat meneruskannya ke parameter `target`:

```python
results = AutoDataMining(
    task="anomaly",
    target="is_fraud",
    models=["isolation_forest", "one_class_svm"],
    random_state=42,
).fit(data)

# Skor Precision, Recall, dan F1 pada holdout test set akan dihitung secara objektif
print(results.metrics)
```

---

## Algoritma Deteksi Anomali yang Didukung

1. `isolation_forest` (Hutan Isolasi berbasis pohon acak; sangat cepat dan efektif untuk data tabular multidimensi).
2. `one_class_svm` (Support Vector Machine Satu Kelas; memetakan batas kerapatan data normal).
3. `lof` / `local_outlier_factor` (Local Outlier Factor; mengukur deviasi kerapatan lokal suatu titik terhadap tetangga terdekatnya).
