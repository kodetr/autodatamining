"""Contoh Regresi Nyata: Prediksi Harga Rumah.

Menunjukkan alur kerja lengkap regresi menggunakan dataset edukasi bawaan:
1. Memuat dataset 'harga_rumah'
2. Menjalankan AutoML regresi (Ridge, Random Forest, Gradient Boosting, dll.)
3. Mengevaluasi metrik (R², MAE, RMSE)
4. Menampilkan kesimpulan naratif otomatis
5. Melakukan hyperparameter tuning otomatis
"""

from autodatamining import AutoDataMining, load_dataset

# 1. Muat dataset edukasi bawaan
data = load_dataset("harga_rumah")
print("=== 5 Baris Pertama Data Harga Rumah ===")
print(data.head())

# 2. Jalankan AutoDataMining untuk regresi
results = AutoDataMining(
    task="regression",
    target="harga_juta",
    scoring="r2",
    cv=5,
    random_state=42,
).fit(data)

# 3. Peringkat Model
print("\n=== Peringkat Model Regresi (Leaderboard) ===")
print(results.leaderboard)

# 4. Metrik Holdout Data Uji
print("\n=== Metrik Data Uji (Holdout) ===")
print(results.metrics)

# 5. Kesimpulan Naratif
print("\n=== Kesimpulan Naratif ===")
print(results.summary())

# 6. Ekspor Tabel LaTeX untuk Publikasi Ilmiah
print("\n=== Kode Tabel LaTeX (Overleaf) ===")
print(results.to_latex())

# 7. Automated Hyperparameter Tuning
print("\n=== Menjalankan Hyperparameter Tuning ===")
tuned = results.tune(method="random", n_iter=6)
print(tuned.summary())
