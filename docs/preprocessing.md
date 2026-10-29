# Flexible preprocessing and diagnostics

AutoDataMining preprocesses features inside each training pipeline. That means imputation,
scaling, and encoding are learned from training data (and separately within each CV fold), which
helps prevent information from validation or test rows leaking into training.

## Mengatur strategi preprocessing

Default untuk kolom numerik adalah imputasi median dan `StandardScaler`. Untuk kolom kategorikal,
default-nya imputasi nilai yang paling sering muncul dan `OneHotEncoder`. Pilihan ini dapat diubah
melalui parameter `preprocessing`:

```python
options = {
    "numeric_imputer": "median",        # mean | median | most_frequent | constant
    "numeric_fill_value": 0,             # dipakai jika numeric_imputer="constant"
    "numeric_scaler": "robust",          # standard | minmax | robust | none
    "categorical_imputer": "constant",   # most_frequent | constant
    "categorical_fill_value": "unknown",
    "categorical_encoder": "ordinal",   # onehot | ordinal
}

results = AutoDataMining(
    task="classification",
    target="label",
    preprocessing=options,
).fit(data)
```

`numeric_scaler=None` atau `"none"` menonaktifkan scaling. Encoder ordinal menggunakan `-1`
untuk kategori yang tidak pernah terlihat saat training. Untuk pilihan `constant`, nilai pengganti
diatur melalui `numeric_fill_value` atau `categorical_fill_value`.

### Transformer kustom dari scikit-learn

Satu transformer dapat menggantikan seluruh preprocessing default. Gunakan ini bila Anda ingin
mengendalikan sendiri pemilihan kolom, imputasi, encoding, atau transformasi:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

features = data.drop(columns=["label"])
numeric_columns = features.select_dtypes(include="number").columns
categorical_columns = features.select_dtypes(exclude="number").columns

custom = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]), numeric_columns),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_columns),
])

results = AutoDataMining(
    task="classification", target="label", preprocessing=custom,
).fit(data)
```

Transformer harus kompatibel dengan scikit-learn dan bisa di-clone. Untuk mengganti hanya satu
cabang, gunakan `numeric_transformer` atau `categorical_transformer` pada dictionary. Opsi
imputer/scaler/encoder di cabang yang sama tidak boleh diberikan bersamaan dengan transformer
kustom cabang tersebut.

Konfigurasi preprocessing yang digunakan disimpan di `results.experiment["preprocessing"]` dan
ikut masuk ke metadata model yang disimpan.

## Grafik diagnostik

Grafik dibuat dari data yang diberikan sebelum preprocessing dan tidak mengubah data atau model.
Semua fungsi mengembalikan `(figure, axes)`; simpan dengan `fig.savefig(...)` atau tampilkan
dengan `show=True`.

```python
from autodatamining import AutoDataMining, plot_preprocessing

fig, axes = AutoDataMining.plot_preprocessing(data, kind="overview", target="label")
fig.savefig("overview.png", dpi=150, bbox_inches="tight")

fig, ax = plot_preprocessing(data, kind="missing_heatmap", target="label")
fig, ax = plot_preprocessing(data, kind="correlation", target="label")
fig, axes = plot_preprocessing(data, kind="distributions", max_features=8)
fig, ax = plot_preprocessing(data, kind="outliers", target="label")
fig, ax = plot_preprocessing(data, kind="target", target="label")
```

Pilihan `kind`:

| `kind` | Isi |
| --- | --- |
| `overview` | Panel ringkasan nilai hilang, korelasi, distribusi fitur, dan target atau fitur kategorikal |
| `missing_heatmap` | Pola nilai hilang per fitur; maksimal 1.000 baris disampling untuk menjaga ukuran grafik |
| `heatmap` | Heatmap korelasi jika ada fitur numerik, atau heatmap nilai hilang jika tidak ada |
| `correlation` | Heatmap korelasi fitur numerik; gagal dengan pesan jelas jika tidak ada fitur numerik |
| `distributions` | Histogram fitur numerik dan frekuensi 10 kategori teratas |
| `outliers` | Boxplot fitur numerik untuk melihat rentang dan nilai ekstrem potensial |
| `target` | Distribusi target; wajib memberikan `target="nama_kolom"` |

Grafik membutuhkan dependency opsional:

```bash
python -m pip install "autodatamining[visualization]"
```

Grafik korelasi, distribusi, dan outlier membantu eksplorasi, tetapi tidak memvalidasi rancangan
split, kesesuaian metrik, kualitas data, atau asumsi penelitian dengan sendirinya.
