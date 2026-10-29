# Panduan Mudah Menggunakan AutoDataMining

Panduan ini menunjukkan alur dasar menggunakan AutoDataMining untuk menyiapkan data,
memeriksa grafik, membandingkan model, membuat prediksi, dan menyimpan model. Contoh utama
menggunakan dataset sintetis supaya bisa dicoba tanpa mengunduh file data.

## 1. Instalasi

Untuk memakai library beserta grafik:

```bash
python -m pip install "autodatamining[visualization]"
```

Untuk pengembangan dari folder source:

```bash
python -m pip install -e ".[visualization]"
```

## 2. Siapkan dan periksa data

Data diberikan sebagai `pandas.DataFrame`. Setiap baris adalah satu observasi dan setiap kolom
adalah fitur atau target. Pada classification dan regression, target adalah kolom yang ingin
diprediksi.

```python
import pandas as pd
from autodatamining import AutoDataMining

data = pd.read_csv("dataset.csv")

# Lihat ukuran data, tipe kolom, nilai kosong, dan kemungkinan masalah.
profile = AutoDataMining.profile(data, target="label")
print(profile["rows"], profile["columns"])
print(profile["column_details"])
print(profile["potential_issues"])
```

Nama `label` pada contoh adalah nama kolom target. Ganti dengan nama kolom target dataset Anda.
Untuk clustering dan anomaly detection tanpa label, tidak perlu menentukan kolom target.

## 3. Lihat grafik sebelum training

Fungsi ini membantu memeriksa nilai kosong, korelasi, distribusi fitur, outlier potensial, dan
distribusi target. Grafik dibuat dari data masukan sebelum preprocessing.

```python
from autodatamining import AutoDataMining

fig, axes = AutoDataMining.plot_preprocessing(
    data,
    kind="overview",
    target="label",
)
fig.savefig("ringkasan-data.png", dpi=150, bbox_inches="tight")

# Pilih grafik lain bila diperlukan.
fig, ax = AutoDataMining.plot_preprocessing(data, kind="missing_heatmap", target="label")
fig.savefig("nilai-kosong.png", dpi=150, bbox_inches="tight")

fig, ax = AutoDataMining.plot_preprocessing(data, kind="correlation", target="label")
fig.savefig("korelasi.png", dpi=150, bbox_inches="tight")
```

Pilihan `kind` lainnya adalah `distributions`, `outliers`, `target`, dan `heatmap`. Grafik `target`
memerlukan argumen `target="nama_kolom"`. Lihat daftar lengkap dan keterangan grafik di
[panduan preprocessing](preprocessing.md).

## 4. Contoh classification lengkap

Contoh ini membuat data latihan dengan fitur numerik, fitur kategori, dan satu nilai numerik yang
kosong. Preprocessing akan mengisi nilai kosong dan mengubah fitur kategori di dalam pipeline.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from autodatamining import AutoDataMining

X, y = make_classification(
    n_samples=180,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    random_state=42,
)
data = pd.DataFrame(X, columns=[f"fitur_{i}" for i in range(X.shape[1])])
data["jenis"] = np.where(data["fitur_0"] > 0, "tinggi", "rendah")
data.loc[0, "fitur_1"] = np.nan
data["label"] = y

preprocessing = {
    "numeric_imputer": "median",
    "numeric_scaler": "robust",
    "categorical_imputer": "most_frequent",
    "categorical_encoder": "onehot",
}

results = AutoDataMining(
    task="classification",
    target="label",
    models=["logistic_regression", "random_forest"],
    test_size=0.2,
    cv=3,
    random_state=42,
    scoring="f1_weighted",
    preprocessing=preprocessing,
).fit(data)

print(results.leaderboard)  # peringkat model dan skor CV
print(results.metrics)      # metrik holdout untuk model terpilih
print(results.classification_report())

results.plot(kind="confusion_matrix", show=True)
results.report("laporan-klasifikasi.html")
results.to_csv("peringkat-model.csv")
```

Untuk dataset CSV milik Anda, ganti pembuatan `X`, `y`, dan `data` dengan `pd.read_csv(...)`, lalu
sesuaikan `target` dengan nama kolom label. `results.predict(...)` menerima DataFrame fitur tanpa
kolom target:

```python
data_baru = pd.read_csv("data-baru.csv")
prediksi = results.predict(data_baru)
print(prediksi)
```

`data-baru.csv` harus memiliki nama kolom fitur yang sama seperti data training. Urutan kolom boleh
berbeda.

### Menggabungkan model tanpa import estimator scikit-learn

Pilih `voting` atau `stacking` pada parameter `models`; tidak perlu mengimpor VotingClassifier atau
StackingClassifier sendiri. AutoDataMining menyiapkan model dasar dan menggabungkan prediksinya di
dalam pipeline.

```python
results = AutoDataMining(
    task="classification",
    target="label",
    models=["voting", "stacking"],
    cv=5,
    random_state=42,
).fit(data)

print(results.leaderboard)
```

Untuk klasifikasi, voting menggunakan soft voting dan stacking mempelajari model akhir untuk
menggabungkan prediksi. Pada regresi, pilihan yang sama memakai VotingRegressor dan
StackingRegressor. Keduanya dibandingkan sebagai kandidat terpisah; model gabungan terbaik dipilih
dengan skor validasi silang.

## 5. Contoh regression

Pakai `task="regression"` saat target berupa nilai kontinu, misalnya harga, suhu, atau durasi.

```python
import pandas as pd
from sklearn.datasets import make_regression
from autodatamining import AutoDataMining

X, y = make_regression(n_samples=180, n_features=6, noise=12, random_state=42)
data = pd.DataFrame(X, columns=[f"fitur_{i}" for i in range(X.shape[1])])
data["harga"] = y

results = AutoDataMining(
    task="regression",
    target="harga",
    models=["ridge", "random_forest"],
    cv=3,
    random_state=42,
).fit(data)

print(results.leaderboard[["model", "cv_score", "mae", "rmse", "r2", "status"]])
results.plot(kind="residuals", show=True)
```

## 6. Contoh clustering

Clustering mencari kelompok pada fitur tanpa membutuhkan target. Contoh ini membuat data yang
memiliki tiga kelompok.

```python
import pandas as pd
from sklearn.datasets import make_blobs
from autodatamining import AutoDataMining

X, _ = make_blobs(n_samples=150, centers=3, random_state=42)
features = pd.DataFrame(X, columns=["ukuran_1", "ukuran_2"])

results = AutoDataMining(
    task="clustering",
    models=["kmeans", "agglomerative"],
    random_state=42,
).fit(features)

print(results.leaderboard)
features["cluster"] = results.labels
print(features.head())
results.plot(kind="clusters", show=True)
```

Skor silhouette, Davies–Bouldin, dan Calinski–Harabasz adalah ukuran internal clustering, bukan
akurasi terhadap label sebenarnya. Jika memakai DBSCAN, label `-1` menunjukkan noise.

## 7. Contoh anomaly detection

Tanpa label anomali, library menandai observasi yang dianggap tidak biasa. Label hasil `-1`
menunjukkan anomali dan `+1` menunjukkan data normal.

```python
import numpy as np
import pandas as pd
from autodatamining import AutoDataMining

rng = np.random.RandomState(42)
normal = rng.normal(size=(140, 2))
anomali = rng.uniform(low=-6, high=6, size=(12, 2))
data = pd.DataFrame(
    np.vstack([normal, anomali]),
    columns=["pengukuran_1", "pengukuran_2"],
)

results = AutoDataMining(
    task="anomaly",
    models=["isolation_forest", "lof"],
    random_state=42,
).fit(data)

print(results.leaderboard[["model", "anomaly_count", "anomaly_rate", "status"]])
data["hasil_deteksi"] = results.labels
print(data[data["hasil_deteksi"] == -1])
results.plot(kind="anomalies", show=True)
```

Tanpa label ground truth, jumlah atau persentase anomali yang ditemukan bukan skor kualitas model.
Jika label tersedia, tambahkan ke DataFrame dan berikan `target="nama_kolom_label"`; gunakan `1`
untuk anomali dan `0` untuk normal. Dalam mode berlabel, `results.labels` berisi prediksi pada
holdout.

## 8. Simpan dan muat model

Pipeline tersimpan mencakup preprocessing yang sudah dipelajari dan model terpilih.

```python
results.save("model.joblib")
```

Muat model saat diperlukan:

```python
import pandas as pd
from autodatamining import load_model

model = load_model("model.joblib")
data_baru = pd.read_csv("data-baru.csv")
prediksi = model.predict(data_baru)
```

Muat hanya file model dari sumber tepercaya karena file joblib/pickle dapat menjalankan kode saat
dibuka. Versi scikit-learn yang berbeda dapat memengaruhi kompatibilitas.

## Memahami hasil

- `results.leaderboard` membandingkan model yang diminta dan menunjukkan status atau error per model.
- `cv_score` digunakan untuk memilih model dari fold training jika cross-validation tersedia.
- `results.metrics` berisi evaluasi holdout model terpilih; holdout tidak dipakai untuk memilih model.
- Skor tinggi pada satu dataset tidak menjamin performa yang sama pada data baru.
- Untuk langkah preprocessing lain atau transformer scikit-learn kustom, baca
  [panduan preprocessing](preprocessing.md).
