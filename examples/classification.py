"""Contoh Klasifikasi Nyata: Prediksi Kelulusan Mahasiswa.

Menunjukkan alur kerja lengkap klasifikasi menggunakan dataset edukasi bawaan:
1. Memuat dataset 'kelulusan_mahasiswa'
2. Profiling ringkas data
3. Menjalankan AutoML klasifikasi (membandingkan berbagai algoritma)
4. Menampilkan leaderboard & kesimpulan naratif otomatis
5. Melakukan hyperparameter tuning otomatis
6. Mengekspor tabel skripsi format baku
"""

from autodatamining import AutoDataMining, load_dataset

# 1. Muat dataset edukasi bawaan
data = load_dataset("kelulusan_mahasiswa")
print("=== 5 Baris Pertama Data ===")
print(data.head())

# 2. Jalankan AutoDataMining untuk klasifikasi
results = AutoDataMining(
    task="classification",
    target="lulus_tepat_waktu",
    scoring="f1_weighted",
    cv=5,
    random_state=42,
).fit(data)

# 3. Peringkat Model (Leaderboard)
print("\n=== Peringkat Model (Leaderboard) ===")
print(results.leaderboard)

# 4. Kesimpulan Naratif Otomatis (Bahasa Indonesia)
print("\n=== Kesimpulan Naratif Otomatis ===")
print(results.summary())

# 5. Tabel Hasil untuk Bab IV Skripsi
print("\n=== Tabel Format Skripsi ===")
print(results.to_academic_table())

# 6. Automated Hyperparameter Tuning pada model terbaik
print("\n=== Menjalankan Hyperparameter Tuning ===")
tuned = results.tune(method="random", n_iter=6)
print(tuned.summary())
print("\nPerbandingan Skor:")
print(tuned.comparison)
