"""Contoh Alur Riset & Skripsi Lengkap (End-to-End Pipeline).

Menunjukkan alur kerja lengkap seorang mahasiswa/peneliti dari awal sampai siap publikasi:
1. Memuat dataset edukasi bawaan ('kelulusan_mahasiswa')
2. Melakukan data profiling
3. Melatih dan membandingkan model dengan AutoDataMining
4. Melakukan hyperparameter tuning otomatis
5. Mengekspor tabel skripsi (Markdown baku & LaTeX Overleaf)
6. Mencetak kesimpulan naratif otomatis (draf Bab IV Skripsi)
7. Menggambar visualisasi alur pipeline data mining
8. Menyimpan model pipeline utuh untuk deployment
"""

from autodatamining import AutoDataMining, explain_concept, load_dataset, load_model

# 1. Pahami konsep jika belum jelas
print("=== Kamus Konsep Data Mining ===")
print(explain_concept("cross_validation"))

# 2. Muat dataset penelitian
data = load_dataset("kelulusan_mahasiswa")
print("\n=== Ringkasan Data Awal ===")
print(f"Jumlah sampel: {len(data)}, Jumlah kolom: {data.shape[1]}")
print(data.head(3))

# 3. Profiling Data
profil = AutoDataMining.profile(data, target="lulus_tepat_waktu")
print(f"Potensi Isu Data: {profil['potential_issues']}")

# 4. Pelatihan & Perbandingan Model (AutoML)
results = AutoDataMining(
    task="classification",
    target="lulus_tepat_waktu",
    scoring="f1_weighted",
    cv=5,
    random_state=42,
).fit(data)

# 5. Leaderboard Model
print("\n=== Peringkat Model (Leaderboard) ===")
print(results.leaderboard)

# 6. Hyperparameter Tuning Otomatis
print("\n=== Mengoptimalkan Model Terbaik (Hyperparameter Tuning) ===")
tuned = results.tune(method="random", n_iter=6)
print(tuned.summary())
print(tuned.comparison)

# 7. Ekspor Hasil untuk Skripsi & Makalah Ilmiah
print("\n=== Tabel Format Bab IV Skripsi ===")
print(results.to_academic_table())

print("\n=== Kode LaTeX untuk Overleaf / Jurnal ===")
print(results.to_latex())

# 8. Kesimpulan Naratif Otomatis
print("\n=== Draf Pembahasan Skripsi Otomatis ===")
print(results.summary())

# 9. Buat Diagram Alur Pipeline (Simpan sebagai gambar)
try:
    fig = results.plot(kind="pipeline", save_path="alur_penelitian.png")
    print("\nDiagram alur pipeline tersimpan di: alur_penelitian.png")
except Exception as e:
    print(f"\nVisualisasi dilewati: {e}")

# 10. Simpan & Muat Model untuk Inferensi Data Baru
results.save("model_skripsi.joblib")
print("Model pipeline tersimpan di: model_skripsi.joblib")

loaded_model = load_model("model_skripsi.joblib")
data_baru = data.drop(columns=["lulus_tepat_waktu"]).head(2)
prediksi = loaded_model.predict(data_baru)
print(f"Prediksi untuk 2 data baru: {prediksi}")
