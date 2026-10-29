# MASTER PROMPT — AutoDataMining: Universal Python Data Mining Library

## 1. Peran

Bertindak sebagai Senior Python Library Engineer, Machine Learning Engineer, AutoML Researcher, dan Open-Source Maintainer yang berpengalaman membangun library Python, merancang API machine learning, mengembangkan sistem evaluasi eksperimen, menulis unit test, serta menerbitkan package ke PyPI.

Tugas Anda adalah mengembangkan library Python bernama **AutoDataMining**, yaitu library yang memungkinkan pengguna menjalankan, membandingkan, mengevaluasi, dan menjelaskan berbagai algoritma data mining menggunakan API yang konsisten.

Jangan berhenti pada rancangan atau contoh kode. Buat proyek lengkap, implementasikan source code, jalankan pengujian, perbaiki bug, dan siapkan distribusi yang dapat diuji sebelum dipublikasikan.

## 2. Identitas library

Gunakan identitas berikut secara konsisten:

- Nama produk: `AutoDataMining`
- Nama distribusi PyPI yang diusulkan: `autodatamining`
- Nama import Python: `autodatamining`
- Class utama: `AutoDataMining`
- Bahasa implementasi: Python
- Lisensi awal: MIT, dengan ketentuan lisensi dan atribusi diperiksa sebelum rilis
- Fondasi algoritma: scikit-learn
- Target pengguna: mahasiswa, dosen, peneliti, data scientist, dan developer

Nama distribusi `autodatamining` harus diperiksa ketersediaannya di PyPI. Jangan menganggap nama tersebut sudah tersedia atau melakukan publikasi tanpa persetujuan pengguna.

## 3. Visi produk

AutoDataMining bertujuan menyederhanakan workflow data mining. Pengguna cukup menyiapkan dataset, menentukan task dan target bila diperlukan, lalu menjalankan satu pemanggilan untuk membandingkan beberapa algoritma yang kompatibel.

Fitur utama:

1. Automatic task detection dengan validasi dan pilihan task eksplisit.
2. Classification dan regression.
3. Clustering dan anomaly detection.
4. Feature selection dan dimensionality reduction.
5. Automatic preprocessing untuk data tabular.
6. Training dan benchmark banyak algoritma.
7. Leaderboard berdasarkan metrik yang dipilih.
8. Pemilihan model terbaik berdasarkan kriteria yang benar.
9. Explainable AI dan feature importance.
10. Visualisasi dan laporan HTML/CSV.
11. Penyimpanan dan pemuatan pipeline model.
12. Custom estimator registry.
13. Reproducible experiment.
14. API sederhana untuk pemula dan API lanjutan untuk peneliti.
15. Integrasi opsional dengan algoritma tambahan.

Jangan mengklaim library menciptakan algoritma machine learning baru. Nilai utama produk adalah integrasi yang konsisten, evaluasi yang tepat, transparansi, extensibility, dan kemudahan penggunaan.

## 4. Contoh API yang diharapkan

### A. Classification

```python
import pandas as pd
from autodatamining import AutoDataMining

data = pd.read_csv("dataset.csv")

dm = AutoDataMining(
    task="classification",
    target="label",
    test_size=0.2,
    cv=5,
    random_state=42,
    scoring="f1_weighted",
)

results = dm.fit(data)

print(results.leaderboard)
print(results.best_model)
print(results.metrics)

results.plot()
results.report("classification-report.html")
results.save("best-model.joblib")
```

### B. Regression

```python
from autodatamining import AutoDataMining

results = AutoDataMining(
    task="regression",
    target="harga",
    cv=5,
    random_state=42,
).fit(data)

print(results.leaderboard)
print(results.best_model)
print(results.metrics)
```

### C. Clustering

```python
results = AutoDataMining(
    task="clustering",
    models=["kmeans", "dbscan", "agglomerative"],
    random_state=42,
).fit(data)

print(results.leaderboard)
print(results.labels)
```

### D. Anomaly detection

```python
results = AutoDataMining(
    task="anomaly",
    models=["isolation_forest", "lof"],
    random_state=42,
).fit(data)

print(results.leaderboard)
print(results.labels)
```

Semua API tersebut adalah spesifikasi yang harus diwujudkan. Jangan mendokumentasikan method atau parameter sebelum implementasinya tersedia dan telah diuji.

## 5. Arsitektur package

Bangun struktur modular seperti berikut:

```text
AutoDataMining/
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── .gitignore
├── docs/
│   ├── installation.md
│   ├── classification.md
│   ├── regression.md
│   ├── clustering.md
│   ├── anomaly_detection.md
│   ├── explainability.md
│   └── custom_estimators.md
├── examples/
│   ├── classification.py
│   ├── regression.py
│   ├── clustering.py
│   └── anomaly_detection.py
├── src/
│   └── autodatamining/
│       ├── __init__.py
│       ├── version.py
│       ├── core.py
│       ├── config.py
│       ├── registry.py
│       ├── detection.py
│       ├── preprocessing.py
│       ├── evaluation.py
│       ├── results.py
│       ├── explainability.py
│       ├── visualization.py
│       ├── reporting.py
│       ├── persistence.py
│       └── exceptions.py
├── tests/
│   ├── test_classification.py
│   ├── test_regression.py
│   ├── test_clustering.py
│   ├── test_anomaly_detection.py
│   ├── test_preprocessing.py
│   ├── test_evaluation.py
│   ├── test_registry.py
│   ├── test_reporting.py
│   └── test_persistence.py
└── .github/
    └── workflows/
        ├── tests.yml
        └── publish.yml
```

Sesuaikan struktur jika diperlukan, tetapi pertahankan pemisahan tanggung jawab dan kemudahan pengujian.

## 6. Algoritma yang harus didukung

### Classification
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Machine

### Regression
- Linear Regression
- Ridge
- Lasso
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor
- KNN Regressor
- Support Vector Regression

### Clustering
- K-Means
- MiniBatch K-Means
- DBSCAN
- Agglomerative Clustering
- Gaussian Mixture Model

### Anomaly detection
- Isolation Forest
- Local Outlier Factor
- One-Class SVM

### Feature selection dan dimensionality reduction
- Variance Threshold
- SelectKBest
- Mutual Information
- PCA
- Truncated SVD jika sesuai dengan representasi data

Gunakan registry algoritma agar model dapat ditambah tanpa mengubah keseluruhan engine.

XGBoost, LightGBM, CatBoost, dan SHAP boleh disediakan sebagai optional dependencies. Jangan menjadikan seluruh dependensi eksternal wajib untuk instalasi dasar.

## 7. Automatic task detection

Rancang deteksi task yang aman dan transparan.

- Jika `task="classification"` atau `task="regression"`, wajib ada target.
- Jika `task="clustering"` atau `task="anomaly"`, target tidak diperlukan.
- Jika `task="auto"` dan target tersedia, gunakan heuristik yang terdokumentasi untuk membedakan classification dan regression.
- Jangan menganggap seluruh kolom numerik pasti regression.
- Jika deteksi ambigu, berikan peringatan atau minta task ditentukan secara eksplisit.
- Jangan menjalankan clustering secara diam-diam hanya karena target tidak tersedia.

Sediakan data profiler untuk memeriksa jumlah baris, kolom, tipe data, missing values, cardinality, distribusi target, dan potensi masalah dataset.

## 8. Standar evaluasi ilmiah

### Classification
Dukung:
- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC bila sesuai

### Regression
Dukung:
- MAE
- MSE
- RMSE
- R²

### Clustering
Dukung:
- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score

### Anomaly detection
Dukung:
- Anomaly rate dan jumlah anomali tanpa label ground truth.
- Precision, Recall, F1, PR-AUC atau ROC-AUC jika label ground truth tersedia dan kondisi evaluasinya memungkinkan.

Persyaratan metodologi:

1. Pisahkan training dan test secara benar.
2. Lakukan preprocessing di dalam pipeline.
3. Fit preprocessing hanya pada data training.
4. Jalankan cross-validation pada training set jika test set digunakan sebagai evaluasi final.
5. Gunakan stratified splitting untuk classification bila memungkinkan.
6. Pastikan metrik dan arah ranking sesuai dengan task.
7. Pisahkan skor cross-validation dari skor holdout test.
8. Hindari pemilihan model berulang berdasarkan test set.
9. Tangani kelas minoritas, dataset kecil, missing values, dan estimator yang gagal.
10. Catat random seed, konfigurasi, versi library, serta informasi split.
11. Jangan mengklaim skor clustering sebagai accuracy.
12. Jangan mengklaim hubungan kausal dari feature importance.

Jelaskan bahwa model dengan skor tertinggi pada satu eksperimen belum tentu menjadi model terbaik untuk semua dataset atau penggunaan nyata.

## 9. Explainability dan visualisasi

Implementasikan:

- Model-native feature importance.
- Permutation importance.
- SHAP sebagai fitur opsional.
- Confusion matrix.
- Classification report.
- ROC curve dan precision-recall curve jika sesuai.
- Residual plot untuk regression.
- Grafik perbandingan algoritma.
- Visualisasi clustering bila relevan.
- Ekspor tabel leaderboard ke CSV.
- Laporan HTML yang berisi konfigurasi, metrik, ranking, durasi training, dan keterbatasan evaluasi.

Jika model tidak mendukung suatu metode explanation, tampilkan pesan yang informatif. Jangan menghasilkan feature importance palsu.

## 10. Model registry dan estimator kustom

Pengguna lanjutan harus dapat menambahkan estimator sendiri.

Rancang API registry dengan ketentuan:
- Nama model unik.
- Validasi kompatibilitas task.
- Pemeriksaan parameter.
- Clone estimator sebelum training.
- Pengelolaan kegagalan per model.
- Pencatatan metrik, waktu training, dan status.
- Integrasi dengan preprocessing pipeline.

Dokumentasikan cara mendaftarkan custom estimator tanpa harus mengubah source code internal library.

## 11. Persistence dan keamanan

Implementasikan penyimpanan pipeline lengkap, termasuk preprocessing.

Pastikan:
- Prediksi data baru menggunakan pipeline yang telah di-fit.
- Urutan dan nama kolom fitur divalidasi.
- Metadata eksperimen disimpan.
- Ketidakcocokan versi ditangani dengan pesan yang jelas.
- Pengguna diberi peringatan bahwa file joblib/pickle dari sumber tidak tepercaya dapat mengeksekusi kode saat dimuat.
- Dataset sensitif tidak otomatis disertakan dalam laporan.

## 12. Unit testing dan CI

Gunakan pytest dengan dataset sintetis dan dataset bawaan scikit-learn.

Uji:
- Semua task.
- Metrik dan ranking.
- Missing values dan data kategorikal.
- Data tidak seimbang.
- Input tidak valid.
- Model registry dan custom estimator.
- Prediksi data baru.
- Explainability.
- Pembuatan laporan.
- Save/load pipeline.
- Reproducibility.
- Penanganan model gagal.
- Kompatibilitas versi Python dan scikit-learn yang didukung.

Jalankan:

```bash
python -m compileall -q src tests
python -m pytest -q
python -m build
python -m twine check dist/*
```

Tambahkan linting, coverage, dan GitHub Actions CI.

Laporkan hasil pengujian aktual. Jangan menyatakan tes, build, atau kompatibilitas berhasil tanpa menjalankannya.

## 13. Packaging PyPI

Konfigurasikan `pyproject.toml` dengan metadata distribusi `autodatamining`, build backend modern, dependensi inti minimal, optional extras, lisensi, README, dan classifiers yang sesuai.

Periksa bahwa:
- Package yang terinstal dapat di-import sebagai `autodatamining`.
- Class dapat diimpor dari `autodatamining import AutoDataMining`.
- Versioning konsisten.
- Source distribution dan wheel berisi file yang diperlukan.
- README dirender dengan benar di PyPI.
- Contoh kode cocok dengan API aktual.
- Nama distribusi tersedia sebelum publikasi.

Siapkan langkah rilis:
1. Audit source code.
2. Jalankan seluruh tes.
3. Build wheel dan source distribution.
4. Validasi artefak dengan Twine.
5. Unggah ke TestPyPI.
6. Uji instalasi dalam virtual environment bersih.
7. Publikasikan ke PyPI dengan API token atau trusted publishing.
8. Verifikasi instalasi dan buat release tag.

Jangan memasukkan token atau kredensial ke dalam source code.

## 14. Dokumentasi README

README harus berisi:
- Identitas dan tujuan AutoDataMining.
- Status versi dan keterbatasan.
- Instalasi.
- Quick start.
- Contoh setiap task.
- Daftar algoritma dan metrik.
- API lanjutan.
- Custom estimator.
- Explainability.
- Reporting.
- Save/load model.
- Metodologi evaluasi.
- Troubleshooting.
- Kontribusi dan lisensi.
- Panduan publikasi PyPI.

Semua contoh README harus dijalankan sebagai bagian dari pengujian atau diverifikasi secara langsung.

## 15. Urutan pengerjaan

**Tahap A — Audit:** periksa source code yang tersedia dan identifikasi masalah.

**Tahap B — Refactoring:** buat arsitektur modular dan API konsisten.

**Tahap C — Implementasi:** selesaikan registry, task engine, evaluasi, explainability, reporting, persistence, dan dokumentasi.

**Tahap D — Testing:** jalankan tes, perbaiki kegagalan, dan uji contoh penggunaan.

**Tahap E — Packaging:** build distribusi, periksa isi wheel, dan uji instalasi lokal.

**Tahap F — Release readiness:** finalisasi README, changelog, lisensi, CI, dan panduan rilis.

Jangan mengganti nama produk di sebagian file saja. Pastikan seluruh import, package directory, metadata, dokumentasi, tes, contoh, dan konfigurasi konsisten menggunakan `AutoDataMining` sebagai nama produk dan `autodatamining` sebagai nama package.

## 16. Hasil akhir yang wajib diserahkan

1. Laporan audit awal.
2. Arsitektur final.
3. Struktur folder lengkap.
4. Source code seluruh modul.
5. Unit tests.
6. README lengkap.
7. Konfigurasi PyPI.
8. Contoh classification, regression, clustering, dan anomaly detection.
9. Hasil pengujian yang benar-benar dijalankan.
10. ZIP proyek lengkap jika lingkungan mendukung pembuatan file.
11. Instruksi instalasi lokal.
12. Instruksi build dan publikasi PyPI.
13. Daftar fitur yang sudah selesai dan yang masih menjadi roadmap.

**Target akhir:** menghasilkan AutoDataMining sebagai library Python data mining yang modular, extensible, reproducible, memiliki evaluasi yang benar, mudah digunakan oleh pemula, dan layak dipelihara sebagai proyek open-source profesional.

Jangan berhenti pada penjelasan konseptual. Implementasikan kode, jalankan pengujian yang tersedia, perbaiki bug, dan nyatakan secara jujur semua keterbatasan yang belum dapat diverifikasi.