# Menggunakan Ensemble Learning

AutoDataMining menyediakan model voting dan stacking bawaan. Pilih nama model melalui parameter
`models`; Anda tidak perlu mengimpor `VotingClassifier`, `StackingClassifier`, atau estimator dasar
dari scikit-learn.

Impor cukup dari AutoDataMining. Siapkan data sebagai DataFrame; contoh di bawah menggunakan file
CSV dengan kolom fitur dan target.

```python
import pandas as pd
from autodatamining import AutoDataMining

data = pd.read_csv("dataset.csv")
```

## Model ensemble bawaan

| Nama pada `models` | Klasifikasi | Regresi |
| --- | --- | --- |
| `voting` | Soft voting dari Logistic Regression, Random Forest, dan Extra Trees | Rata-rata prediksi Ridge, Random Forest, dan Extra Trees |
| `stacking` | Model akhir Logistic Regression menggabungkan prediksi tiga model dasar | Model akhir Ridge menggabungkan prediksi model dasar |

Kedua teknik membungkus model dasar dalam satu estimator. AutoDataMining kemudian mengevaluasi
estimator gabungan tersebut sebagai kandidat model. Bila Anda memasukkan keduanya, hasil leaderboard
akan membandingkan voting dengan stacking serta estimator lain yang Anda pilih.

## Klasifikasi

Pastikan tabel memiliki kolom target, misalnya `label`. `data` pada contoh ini adalah DataFrame yang
sudah dibaca dan disiapkan.

```python
results = AutoDataMining(
    task="classification",
    target="label",
    models=["voting", "stacking"],
    scoring="f1_weighted",
    cv=5,
    random_state=42,
).fit(data)

print(results.leaderboard[["model", "cv_score", "status"]])
print("Model terpilih:", results.model_name)
print(results.metrics)
```

Gunakan hanya satu teknik bila ingin menguji teknik itu saja:

```python
results = AutoDataMining(
    task="classification",
    target="label",
    models=["voting"],
    cv=5,
    random_state=42,
).fit(data)
```

Voting menggunakan **soft voting**: model dasar memberi probabilitas kelas dan AutoDataMining
menggabungkan probabilitas tersebut. Stacking menggunakan prediksi model dasar untuk melatih model
akhir Logistic Regression. Validasi silang luar (`cv`) digunakan AutoDataMining untuk memilih
kandidat; stacking juga memakai validasi internal tiga fold untuk melatih model akhirnya.

## Regresi

Target regresi harus berupa nilai numerik kontinu. Nama `voting` dan `stacking` otomatis mengarah ke
varian regresi.

```python
results = AutoDataMining(
    task="regression",
    target="harga",
    models=["voting", "stacking"],
    cv=5,
    random_state=42,
).fit(data)

print(results.leaderboard[["model", "cv_score", "mae", "rmse", "r2", "status"]])
print("Model terpilih:", results.model_name)
```

VotingRegressor merata-ratakan prediksi Ridge, Random Forest, dan Extra Trees. StackingRegressor
menggunakan Ridge sebagai model akhir yang belajar dari prediksi model dasar.

## Mengatur model dasar

Anda dapat mengubah parameter estimator dasar tanpa mengimpor estimator scikit-learn. Gunakan nama
parameter bertingkat pada `model_params`:

```python
results = AutoDataMining(
    task="classification",
    target="label",
    models=["voting", "stacking"],
    model_params={
        "voting": {
            "random_forest__n_estimators": 200,
            "extra_trees__n_estimators": 200,
        },
        "stacking": {
            "random_forest__n_estimators": 150,
            "extra_trees__n_estimators": 150,
        },
    },
    cv=5,
    random_state=42,
).fit(data)
```

Nama sebelum `__` adalah nama model dasar di dalam ensemble. `random_state` eksperimen diteruskan
ke estimator dasar yang memiliki parameter tersebut, sehingga proses lebih mudah direproduksi.

## Membaca hasil dan batasan

- `results.leaderboard` menampilkan skor cross-validation, status, dan model yang dibandingkan.
- `results.model_name` menunjukkan model dengan skor CV terbaik ketika skor tersebut tersedia.
- `results.metrics` berisi metrik holdout untuk model terpilih; holdout tidak digunakan memilih model.
- Stacking melakukan pelatihan internal tambahan dan umumnya lebih lambat daripada voting.
- Jika jumlah data pada suatu kelas terlalu sedikit untuk validasi internal stacking, pilih voting
  atau tambah data untuk kelas tersebut.
- Model gabungan belum tentu lebih baik daripada model tunggal. Bandingkan hasilnya dengan metrik
  yang sesuai dan evaluasi pada data yang belum digunakan saat training.

Untuk contoh alur data lengkap, lihat [Panduan Mudah Menggunakan AutoDataMining](panduan-penggunaan.md).
