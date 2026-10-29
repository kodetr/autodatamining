"""Contoh Clustering Nyata: Segmentasi Profil Nasabah Bank.

Menunjukkan alur kerja unsupervised clustering menggunakan dataset edukasi bawaan:
1. Memuat dataset 'segmentasi_nasabah'
2. Menjalankan AutoML clustering (K-Means, DBSCAN, Agglomerative)
3. Membandingkan skor Silhouette
4. Menganalisis karakteristik tiap cluster nasabah
"""

import pandas as pd
from autodatamining import AutoDataMining, load_dataset

# 1. Muat dataset edukasi bawaan
data = load_dataset("segmentasi_nasabah")
print("=== 5 Baris Pertama Data Nasabah ===")
print(data.head())

# 2. Jalankan AutoDataMining untuk clustering
results = AutoDataMining(
    task="clustering",
    models=["kmeans", "agglomerative", "dbscan"],
    random_state=42,
).fit(data)

# 3. Peringkat Model Berdasarkan Silhouette Score
print("\n=== Peringkat Model Clustering ===")
print(results.leaderboard)

# 4. Tambahkan label cluster ke data asli untuk interpretasi bisnis
data_cluster = data.copy()
data_cluster["cluster"] = results.labels
print("\n=== Rata-rata Profil Tiap Kelompok Nasabah ===")
print(data_cluster.groupby("cluster").mean().round(2))

# 5. Kesimpulan Naratif
print("\n=== Kesimpulan Naratif ===")
print(results.summary())
