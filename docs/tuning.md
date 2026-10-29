# Panduan Penyetelan Hyperparameter Otomatis (`results.tune()`)

Penyetelan hyperparameter (*hyperparameter tuning*) adalah proses mencari kombinasi parameter model yang paling optimal guna memaksimalkan akurasi atau meminimalkan galat prediksi.

Di **AutoDataMining**, fungsi `results.tune()` dirancang agar:
1. **Otomatis:** Langsung menyetel model pemenang dari leaderboard tanpa perlu menulis ratusan baris kode `GridSearchCV`.
2. **Bebas Kebocoran Data (*Zero Data Leakage*):** Penyetelan dilakukan murni di dalam lipatan data latih (*train folds*), dan performa akhir diuji pada *test holdout* yang independen.
3. **Analitis:** Menyediakan perbandingan sebelum vs sesudah tuning serta draf narasi ilmiah.

---

## Contoh Penggunaan Cepat

```python
from autodatamining import AutoDataMining, load_dataset

# 1. Latih model klasifikasi
data = load_dataset("kelulusan_mahasiswa")
results = AutoDataMining(task="classification", target="lulus_tepat_waktu").fit(data)

# 2. Jalankan tuning otomatis (metode acak / random search)
tuned = results.tune(method="random", n_iter=8, cv=3)

# 3. Cetak narasi evaluasi sebelum vs sesudah
print(tuned.summary())

# 4. Tabel perbandingan skor
print(tuned.comparison)

# 5. Parameter terbaik yang ditemukan
print("Parameter optimal:", tuned.best_params)

# 6. Prediksi data baru menggunakan pipeline optimal
y_pred = tuned.predict(data_baru)
```

---

## Metode Penyetelan: `random` vs `grid`

Parameter `method` mendukung dua strategi:
1. `method="random"` (Bawaan):
   - Menggunakan `RandomizedSearchCV`.
   - Mengambil sampel kombinasi secara acak sebanyak `n_iter` kali.
   - Jauh lebih cepat dan efisien untuk model dengan banyak parameter seperti Random Forest atau Gradient Boosting.
2. `method="grid"`:
   - Menggunakan `GridSearchCV`.
   - Menguji seluruh kemungkinan kombinasi parameter secara tuntas. Cocok untuk ruang pencarian kecil.

```python
# Menggunakan Grid Search tuntas
tuned_grid = results.tune(method="grid", cv=5)
```

---

## Ruang Pencarian Default (*Default Search Spaces*)

AutoDataMining telah menyediakan ruang pencarian default yang teruji untuk setiap algoritma:

| Model | Parameter yang Disetel |
| --- | --- |
| `random_forest` | `n_estimators`, `max_depth`, `min_samples_split` |
| `gradient_boosting` | `learning_rate`, `n_estimators`, `max_depth` |
| `hist_gradient_boosting` | `learning_rate`, `max_iter` |
| `decision_tree` | `max_depth`, `min_samples_split` |
| `logistic_regression` | `C` (parameter regularisasi) |
| `linear_regression` | `fit_intercept`, `positive` |
| `ridge` / `lasso` | `alpha` (kekuatan penalti regularisasi) |
| `knn` | `n_neighbors`, `weights` (`uniform` / `distance`) |
| `svm` | `C`, `gamma` |
| `gaussian_nb` | `var_smoothing` |

---

## Kustomisasi Ruang Parameter (`param_grid`)

Jika Anda ingin menentukan rentang parameter sendiri:

```python
custom_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [5, 10, None],
}

tuned = results.tune(param_grid=custom_grid, method="grid")
```
