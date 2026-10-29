"""Contoh Deteksi Anomali Nyata: Deteksi Transaksi Mencurigakan (Fraud).

Menunjukkan alur deteksi anomali tanpa label (unsupervised) menggunakan dataset bawaan:
1. Memuat dataset 'deteksi_penipuan'
2. Menjalankan AutoML deteksi anomali (Isolation Forest, One-Class SVM, LOF)
3. Mendeteksi dan mengisolasi transaksi anomali (-1)
4. Menganalisis perbedaan karakteristik transaksi normal vs anomali
"""

import pandas as pd
from autodatamining import AutoDataMining, load_dataset

# 1. Muat dataset edukasi bawaan
data = load_dataset("deteksi_penipuan")
print("=== 5 Baris Pertama Data Transaksi ===")
print(data.head())

# 2. Jalankan AutoDataMining untuk deteksi anomali
results = AutoDataMining(
    task="anomaly",
    models=["isolation_forest", "one_class_svm", "local_outlier_factor"],
    random_state=42,
).fit(data)

# 3. Leaderboard Deteksi Anomali
print("\n=== Ringkasan Deteksi Anomali ===")
print(results.leaderboard)

# 4. Tambahkan label prediksi (-1: Anomali/Mencurigakan, 1: Normal)
data_hasil = data.copy()
data_hasil["status"] = ["Anomali" if lbl == -1 else "Normal" for lbl in results.labels]

print("\n=== Jumlah Deteksi ===")
print(data_hasil["status"].value_counts())

print("\n=== Contoh Transaksi yang Terdeteksi Sebagai Anomali ===")
anomali = data_hasil[data_hasil["status"] == "Anomali"]
print(anomali.head(10))

# 5. Kesimpulan Naratif
print("\n=== Kesimpulan Naratif ===")
print(results.summary())
