# Panduan Pengelompokan (Clustering)

Modul clustering di **AutoDataMining** digunakan untuk menemukan struktur dan segmentasi kelompok alami dalam data tanpa memerlukan label target (*unsupervised learning*).

---

## Contoh Cepat dengan Dataset Nyata

Gunakan dataset bawaan `"segmentasi_nasabah"` untuk mengelompokkan profil finansial nasabah perbankan:

```python
import pandas as pd
from autodatamining import AutoDataMining, load_dataset

# 1. Muat dataset edukasi nasabah bank
data = load_dataset("segmentasi_nasabah")
print(data.head())

# 2. Jalankan AutoML Clustering
results = AutoDataMining(
    task="clustering",
    models=["kmeans", "agglomerative", "dbscan"],
    random_state=42,
).fit(data)

# 3. Peringkat Model Berdasarkan Silhouette Score
print(results.leaderboard)

# 4. Tambahkan Label Klaster ke Data Asli
data_hasil = data.copy()
data_hasil["klaster"] = results.labels

# 5. Analisis Profil Karakteristik Tiap Klaster (Insight Bisnis)
print("\nRata-rata Tiap Kelompok Nasabah:")
print(data_hasil.groupby("klaster").mean().round(2))

# 6. Kesimpulan Naratif Otomatis
print(results.summary())

# 7. Visualisasi Klaster dalam Ruang 2D (PCA)
results.plot(kind="clusters", save_path="sebaran_klaster.png")
```

---

## Algoritma Clustering yang Didukung

1. `kmeans` (K-Means Clustering)
2. `agglomerative` (Hierarchical Agglomerative Clustering)
3. `dbscan` (Density-Based Spatial Clustering of Applications with Noise)
4. `minibatch_kmeans` (Mini-Batch K-Means untuk dataset berskala besar)
5. `gaussian_mixture` (Gaussian Mixture Models / GMM)

---

## Cara Membaca Metrik Clustering

- **Silhouette Score:** 
  - Rentang nilai dari $-1.0$ hingga $+1.0$.
  - Skor $> 0.5$ menandakan klaster terbentuk dengan pemisahan yang tegas dan kompak.
  - Skor mendekati $0$ menandakan adanya tumpang-tindih batas antarkelompok.
- **Davies-Bouldin Index:** Mengukur kemiripan antar-klaster (semakin kecil mendekati 0, semakin baik kualitas pemisahannya).
- **Calinski-Harabasz Index:** Rasio varians antar-klaster terhadap varians di dalam klaster (semakin tinggi skor, semakin kompak klaster).
- **Noise Count:** Khusus algoritma `dbscan`, mencatat jumlah observasi pencilan yang tidak masuk ke klaster manapun (diberi label `-1`).
