# Panduan Regresi (Regression)

Modul regresi di **AutoDataMining** digunakan untuk memperkirakan variabel kontinu (*angka numerik riil*) seperti harga properti, estimasi pendapatan, atau perkiraan konsumsi energi.

---

## Contoh Cepat dengan Dataset Nyata

Gunakan dataset bawaan `"harga_rumah"` untuk memprediksi harga jual rumah (dalam Juta Rupiah):

```python
from autodatamining import AutoDataMining, load_dataset

# 1. Muat dataset edukasi harga rumah
data = load_dataset("harga_rumah")
print(data.head())

# 2. Latih dan bandingkan model regresi
results = AutoDataMining(
    task="regression",
    target="harga_juta",
    scoring="r2",
    cv=5,
    random_state=42,
).fit(data)

# 3. Peringkat Model (Leaderboard)
print(results.leaderboard)

# 4. Metrik Holdout Data Uji (R², MAE, MSE, RMSE)
print("Metrik Evaluasi:", results.metrics)

# 5. Kesimpulan Naratif Otomatis
print(results.summary())

# 6. Tabel Skripsi & Kode LaTeX
print(results.to_academic_table())
print(results.to_latex())

# 7. Visualisasi Residuals & Diagram Pipeline
results.plot(kind="residuals", save_path="plot_residual.png")
results.plot(kind="pipeline", save_path="pipeline_regresi.png")
```

---

## Algoritma Regresi yang Didukung

AutoDataMining membandingkan 12 model regresi terkemuka:
1. `linear_regression` (Ordinary Least Squares)
2. `ridge` (Regresi Ridge / L2 Regularization)
3. `lasso` (Regresi Lasso / L1 Regularization)
4. `decision_tree` (Pohon Keputusan Regresi)
5. `random_forest` (Hutan Acak Regresi)
6. `extra_trees` (Extremely Randomized Trees Regressor)
7. `gradient_boosting` (Gradient Boosting Regressor)
8. `hist_gradient_boosting` (Histogram-based Gradient Boosting)
9. `knn` (K-Nearest Neighbors Regressor)
10. `svm` (Support Vector Regressor / SVR)
11. `voting` (Voting Regressor)
12. `stacking` (Stacking Meta-Regressor)

---

## Cara Membaca Metrik Regresi

1. **R² (Koefisien Determinasi):** 
   - Nilai berkisar antara $0.0$ hingga $1.0$ (dapat negatif jika model lebih buruk daripada garis rata-rata).
   - Menunjukkan persentase variasi nilai target yang berhasil dijelaskan oleh fitur.
2. **MAE (Mean Absolute Error):** 
   - Rata-rata selisih absolut antara prediksi dan nilai riil. Interpretasinya lugas karena menggunakan satuan yang sama dengan variabel target (misalnya: selisih Rp 42 Juta).
3. **RMSE (Root Mean Squared Error):** 
   - Akar kuadrat dari rata-rata galat kuadrat. Memberikan penalti lebih berat pada kesalahan prediksi yang berjarak sangat jauh (outlier).
