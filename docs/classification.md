# Panduan Klasifikasi (Classification)

Modul klasifikasi di **AutoDataMining** digunakan untuk memprediksi variabel target kategori (*kategorikal / diskrit*) dari sekumpulan fitur numerik dan kategorik.

---

## Contoh Cepat dengan Dataset Nyata

Gunakan dataset bawaan `"kelulusan_mahasiswa"` untuk memprediksi apakah seorang mahasiswa lulus tepat waktu:

```python
from autodatamining import AutoDataMining, load_dataset

# 1. Muat dataset edukasi
data = load_dataset("kelulusan_mahasiswa")
print(data.head())

# 2. Latih dan bandingkan semua algoritma klasifikasi
results = AutoDataMining(
    task="classification",
    target="lulus_tepat_waktu",
    scoring="f1_weighted",
    cv=5,
    random_state=42,
).fit(data)

# 3. Peringkat Model (Leaderboard)
print(results.leaderboard)

# 4. Ringkasan & Kesimpulan Naratif
print(results.summary())

# 5. Tabel Skripsi & Kode LaTeX
print(results.to_academic_table())
print(results.to_latex())

# 6. Visualisasi
results.plot(kind="pipeline", save_path="pipeline_klasifikasi.png")
results.plot(kind="confusion_matrix", save_path="matriks_kebingungan.png")
results.plot(kind="roc_curve", save_path="kurva_roc.png")
```

---

## Algoritma Klasifikasi yang Didukung

Secara default, AutoDataMining membandingkan 11 model klasifikasi:
1. `logistic_regression` (Regresi Logistik)
2. `decision_tree` (Pohon Keputusan / CART)
3. `random_forest` (Hutan Acak Ensemble Bagging)
4. `extra_trees` (Extremely Randomized Trees)
5. `gradient_boosting` (Ensemble Boosting)
6. `hist_gradient_boosting` (Histogram-based Boosting)
7. `knn` (K-Nearest Neighbors)
8. `gaussian_nb` (Naive Bayes Gaussian)
9. `svm` (Support Vector Machine / SVC)
10. `voting` (Ensemble Voting / Soft Voting)
11. `stacking` (Ensemble Stacking Meta-Classifier)

Anda dapat memilih algoritma tertentu menggunakan parameter `models`:
```python
results = AutoDataMining(
    task="classification",
    target="lulus_tepat_waktu",
    models=["random_forest", "gradient_boosting", "svm"],
).fit(data)
```

---

## Metrik Evaluasi yang Dilaporkan

AutoDataMining mengevaluasi performa model menggunakan metrik terstandar:
- **Accuracy:** Persentase prediksi benar terhadap seluruh data.
- **Balanced Accuracy:** Rata-rata akurasi pada tiap kelas (sangat berguna untuk dataset timpang/imbalanced).
- **Precision (Weighted):** Ketepatan model saat memprediksi kelas positif.
- **Recall (Weighted):** Kemampuan model menemukan seluruh kasus positif riil.
- **F1-Score (Weighted):** Nilai harmonic mean antara precision dan recall.
- **ROC-AUC & PR-AUC:** Luas area di bawah kurva karakteristik operasi penerima (untuk klasifikasi biner).

---

## Optimasi Hyperparameter (`results.tune()`)

Model pemenang leaderboard dapat ditingkatkan performanya secara otomatis:
```python
tuned = results.tune(method="random", n_iter=8, cv=3)
print(tuned.summary())
print("Parameter optimal:", tuned.best_params)

# Prediksi data baru dengan model optimal
prediksi = tuned.predict(data_baru)
```
