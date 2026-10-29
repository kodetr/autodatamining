# AutoDataMining
**AutoDataMining** adalah pustaka Python untuk membantu menjalankan alur kerja data mining pada data berbentuk tabel dengan scikit-learn. Pustaka ini menyatukan prapemrosesan, perbandingan model, evaluasi, visualisasi, laporan, dan penyimpanan model dalam satu antarmuka.

Dengan AutoDataMining, pengguna dapat memilih jenis tugas, menentukan kolom target bila diperlukan, membandingkan beberapa estimator, melihat metrik evaluasi, lalu menyimpan pipeline yang mencakup prapemrosesan dan model.

AutoDataMining mengintegrasikan algoritma yang tersedia di scikit-learn dan tidak menciptakan algoritma machine learning baru. Termasuk di dalamnya teknik ensemble seperti voting dan stacking untuk menggabungkan prediksi beberapa model. Hasil bergantung pada dataset, cara pembagian data, metrik, dan parameter yang digunakan; skor tinggi pada satu percobaan bukan jaminan performa yang sama pada data lain.

> **Versi 0.2.0.** Pustaka siap pakai untuk pendidikan, riset, dan publikasi ilmiah dengan dukungan No-Code GUI di Hugging Face Spaces.

## Fitur utama

- Klasifikasi dan regresi dengan pemisahan data pelatihan dan data uji.
- Pengelompokan data (clustering) dan deteksi anomali.
- Penambangan aturan asosiasi (Association Rule Mining) dengan Apriori dan FP-Growth untuk analisis transaksi.
- **Antarmuka Web Dashboard No-Code (GUI Interaktif)** berbasis Streamlit untuk Hugging Face Spaces dan penggunaan lokal (`python -m autodatamining.ui`).
- Dataset edukasi bawaan (load_dataset) untuk pembelajaran instan tanpa perlu mencari atau mengunduh file CSV.
- Modul praktikum interaktif Google Colab / Jupyter Notebooks di folder `notebooks/` untuk seluruh 5 pilar data mining.
- Penyetelan hyperparameter otomatis (results.tune) untuk mengoptimalkan model terbaik.
- Diagram alur pipeline visual (results.plot("pipeline")) untuk slide sidang skripsi dan publikasi.
- Kesimpulan naratif otomatis (results.summary) dan ekspor tabel format skripsi / LaTeX (to_academic_table, to_latex).
- Kamus konsep interaktif (explain_concept) untuk memahami konsep penting data mining.
- Prapemrosesan numerik dan kategorikal yang dapat dikonfigurasi.
- Perbandingan model scikit-learn (estimator) melalui tabel peringkat.
- Ensemble bawaan dengan voting dan stacking untuk klasifikasi serta regresi.
- Validasi silang pada data pelatihan dan metrik data uji yang dilaporkan terpisah.
- Profil data, grafik diagnostik, penjelasan model dasar, dan laporan HTML.
- Penyimpanan pipeline lengkap agar prapemrosesan yang sama digunakan saat prediksi.
- Pendaftaran estimator kustom.

## Instalasi

Instalasi standar:

~~~bash
python -m pip install autodatamining
~~~

Untuk memakai grafik dan visualisasi hasil:

~~~bash
python -m pip install "autodatamining[visualization]"
~~~

Untuk menjalankan Web Dashboard No-Code (GUI) secara lokal:

~~~bash
python -m pip install "autodatamining[ui]"
python -m autodatamining.ui
~~~

Atau coba langsung secara online tanpa install di Hugging Face Spaces:
👉 **[https://huggingface.co/spaces/kodetr/autodatamining](https://huggingface.co/spaces/kodetr/autodatamining)**

Untuk panduan deploy sendiri ke Space Anda, baca [Panduan Hugging Face Spaces](docs/huggingface-spaces.md).

Untuk fitur SHAP opsional:

~~~bash
python -m pip install "autodatamining[explainability]"
~~~

## Cara penggunaan

Contoh berikut membaca data (bisa menggunakan file CSV Anda sendiri atau `load_dataset` bawaan), memeriksa ringkasan data, melihat grafik diagnostik, lalu membandingkan model klasifikasi (classification):

~~~python
import pandas as pd
from autodatamining import AutoDataMining, load_dataset, explain_concept

# Gunakan dataset edukasi bawaan atau baca dari pd.read_csv("dataset.csv")
data = load_dataset("kelulusan_mahasiswa")

# Periksa jumlah baris/kolom, tipe data, nilai kosong, dan target.
profile = AutoDataMining.profile(data, target="lulus_tepat_waktu")
print(profile["potential_issues"])

# Grafik diagnostik sebelum pelatihan. Perlu extra visualization.
fig, axes = AutoDataMining.plot_preprocessing(
    data, kind="overview", target="lulus_tepat_waktu"
)
fig.savefig("ringkasan-data.png", dpi=150, bbox_inches="tight")

# Konfigurasi ini opsional. Pengaturan bawaan akan dipakai bila tidak diberikan.
preprocessing = {
    "numeric_imputer": "median",
    "numeric_scaler": "robust",
    "categorical_imputer": "most_frequent",
    "categorical_encoder": "onehot",
}

results = AutoDataMining(
    task="classification",
    target="lulus_tepat_waktu",
    models=["logistic_regression", "random_forest"],
    test_size=0.2,
    cv=5,
    random_state=42,
    scoring="f1_weighted",
    preprocessing=preprocessing,
).fit(data)

print(results.leaderboard)        # Peringkat model dan skor validasi silang
print(results.summary())          # Kesimpulan naratif otomatis & draf pembahasan skripsi
print(results.to_academic_table())# Tabel format baku Bab IV Skripsi / Jurnal
print(results.to_latex())         # Kode tabel LaTeX siap pakai di Overleaf
print(results.metrics)            # Metrik data uji untuk model terpilih

# Penyetelan hyperparameter otomatis model terbaik
tuned = results.tune()
print(tuned.summary())            # Evaluasi Sebelum vs Sesudah Tuning

# Tampilkan diagram alur pipeline visual dan matriks kebingungan
results.plot(kind="pipeline", show=True)
results.plot(kind="confusion_matrix", show=True)
results.report("laporan.html")
results.to_csv("leaderboard.csv")
results.save("model.joblib")
~~~

Untuk memprediksi data baru, kolom fiturnya harus sama dengan data training. Urutan kolom boleh berbeda.

~~~python
import pandas as pd
from autodatamining import load_model

model = load_model("model.joblib")
data_baru = pd.read_csv("data-baru.csv")
prediksi = model.predict(data_baru)
print(prediksi)
~~~

Untuk panduan pemula dengan contoh lengkap yang dapat dijalankan untuk semua jenis tugas, buka [Panduan Mudah Menggunakan AutoDataMining](docs/panduan-penggunaan.md).

## Jenis tugas

| Tugas | Kapan digunakan | Perlu target? |
| --- | --- | --- |
| Klasifikasi (classification) | Memprediksi kelas/kategori, misalnya status pelanggan | Ya |
| Regresi (regression) | Memprediksi nilai kontinu, misalnya harga | Ya |
| Pengelompokan (clustering) | Mengelompokkan data tanpa label kelas | Tidak |
| Anomali (anomaly) | Menemukan observasi yang tidak biasa | Tidak; label opsional untuk evaluasi |
| Aturan Asosiasi (association) | Menemukan pola kemunculan bersama (misal: analisis keranjang belanja) | Tidak |

Tentukan jenis tugas secara eksplisit bila tujuan analisis sudah diketahui. Mode otomatis hanya memilih klasifikasi atau regresi dari kolom target dan tidak akan memulai pengelompokan secara diam-diam.

## Contoh tugas lainnya

### Regresi

~~~python
results = AutoDataMining(
    task="regression",
    target="price",
    models=["ridge", "random_forest", "hist_gradient_boosting"],
    cv=5,
    random_state=42,
).fit(data)

print(results.leaderboard[["model", "cv_score", "mae", "rmse", "r2", "status"]])
results.plot(kind="residuals", show=True)
~~~

### Pengelompokan data

~~~python
results = AutoDataMining(
    task="clustering",
    models=["kmeans", "dbscan", "agglomerative"],
    random_state=42,
).fit(features)

print(results.leaderboard)
print(results.labels)  # Label cluster untuk setiap baris
results.plot(kind="clusters", show=True)
~~~

Silhouette, Davies–Bouldin, dan Calinski–Harabasz adalah metrik internal pengelompokan, bukan ukuran akurasi terhadap kelas yang diketahui. Pada DBSCAN, label -1 menunjukkan data noise.

### Deteksi anomali

~~~python
results = AutoDataMining(
    task="anomaly",
    models=["isolation_forest", "lof"],
    random_state=42,
).fit(features)

print(results.leaderboard)  # anomaly_count dan anomaly_rate
print(results.labels)       # -1 = anomali, +1 = data normal
results.plot(kind="anomalies", show=True)
~~~

Tanpa label acuan (ground truth), jumlah atau persentase anomali yang ditemukan tidak dapat digunakan untuk menentukan kualitas model. Jika label tersedia, berikan nama kolomnya melalui target dan gunakan 1/True untuk anomali serta 0/False untuk data normal. Dalam mode berlabel, prediksi yang dikembalikan adalah prediksi pada data uji.

### Aturan Asosiasi

~~~python
results = AutoDataMining(
    task="association",
    min_support=0.3,
    min_confidence=0.6,
).fit(transaksi)

print(results.leaderboard)        # Perbandingan kecepatan FP-Growth vs Apriori
print(results.frequent_itemsets) # Pola itemset yang sering muncul
print(results.rules)             # Aturan antecedents -> consequents, support, confidence, lift
results.plot(kind="rules", show=True)
~~~

Lihat [panduan aturan asosiasi](docs/aturan-asosiasi.md) untuk rumus lengkap dan contoh penggunaan standalone.

## Menggabungkan model dengan ensemble

Pilih model gabungan langsung melalui parameter models. Tidak perlu mengimpor VotingClassifier atau StackingClassifier dari scikit-learn.

~~~python
results = AutoDataMining(
    task="classification",
    target="label",
    models=["voting", "stacking"],
    cv=5,
    random_state=42,
).fit(data)

print(results.leaderboard)
print(results.best_model)
~~~

Untuk klasifikasi, voting menggabungkan probabilitas prediksi (soft voting), sedangkan stacking melatih model akhir untuk mempelajari cara menggabungkan prediksi model dasar. Untuk regresi, nama voting dan stacking menggunakan VotingRegressor dan StackingRegressor. Keduanya dievaluasi sebagai kandidat model; AutoDataMining memilih hasil terbaik berdasarkan skor validasi silang.

Lihat [panduan penggunaan Ensemble Learning](docs/ensemble-learning.md) untuk contoh klasifikasi, regresi, dan pengaturan model dasar tanpa import estimator scikit-learn.

## Algoritma dan metrik

| Tugas | Estimator bawaan | Skor perbandingan bawaan |
| --- | --- | --- |
| Klasifikasi | Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, KNN, Gaussian Naive Bayes, SVM, voting, stacking | Accuracy pada validasi silang data pelatihan |
| Regresi | Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, KNN, SVR, voting, stacking | R² pada validasi silang data pelatihan |
| Pengelompokan | K-Means, MiniBatch K-Means, DBSCAN, Agglomerative, Gaussian Mixture | Silhouette |
| Anomali | Isolation Forest, Local Outlier Factor, One-Class SVM | F1 validasi silang jika label tersedia; tanpa label tidak ada peringkat kualitas |
| Aturan Asosiasi | FP-Growth, Apriori | Kecepatan eksekusi (detik) dan jumlah aturan yang dihasilkan |

Metrik klasifikasi pada data uji mencakup accuracy, balanced accuracy, precision, recall, dan F1 berbobot. ROC-AUC dan PR-AUC ditampilkan bila model dan data mendukungnya. Regresi melaporkan MAE, MSE, RMSE, dan R². Metrik pengelompokan hanya ditampilkan bila dapat dihitung.

Jika satu estimator gagal, status dan pesan kesalahannya dicatat di tabel peringkat; model lain tetap dapat digunakan.

## Profil data dan deteksi tugas

Profil data membantu melihat ukuran dataset, tipe kolom, nilai kosong, jumlah nilai unik, duplikasi baris, dan ringkasan target.

~~~python
profile = AutoDataMining.profile(data, target="label")
print(profile["column_details"])
print(profile["potential_issues"])
~~~

Dengan deteksi otomatis, target teks diperlakukan sebagai klasifikasi. Target integer dengan jumlah kelas rendah juga dapat dianggap sebagai klasifikasi; target numerik lainnya biasanya dianggap regresi. Jika interpretasi target numerik tidak jelas, tetapkan jenis tugas secara eksplisit.

## Prapemrosesan yang fleksibel

Pengaturan prapemrosesan bawaan:

- Kolom numerik: imputasi median dan StandardScaler.
- Kolom kategorikal: imputasi nilai paling sering dan OneHotEncoder.

Ubah strategi per eksperimen:

~~~python
preprocessing = {
    "numeric_imputer": "median",        # mean, median, most_frequent, constant
    "numeric_fill_value": 0,             # digunakan jika imputer constant
    "numeric_scaler": "robust",          # standard, minmax, robust, none
    "categorical_imputer": "constant",   # most_frequent atau constant
    "categorical_fill_value": "unknown",
    "categorical_encoder": "ordinal",   # onehot atau ordinal
}

results = AutoDataMining(
    task="classification",
    target="label",
    preprocessing=preprocessing,
).fit(data)
~~~

Prapemrosesan dipelajari di dalam pipeline pelatihan dan setiap fold validasi silang, sehingga statistik imputasi dan scaling tidak mengambil informasi dari fold validasi atau data uji.

Pengguna lanjutan dapat memberikan transformer scikit-learn yang bisa di-clone melalui parameter preprocessing untuk mengganti seluruh preprocessing bawaan. Transformer numerik atau kategorikal juga dapat diganti secara terpisah melalui opsi numeric_transformer atau categorical_transformer. Baca [panduan preprocessing](docs/preprocessing.md) untuk contoh lengkap.

## Grafik diagnostik

Grafik diagnostik memeriksa data masukan sebelum transformasi dan tidak mengubah model.

~~~python
fig, axes = AutoDataMining.plot_preprocessing(
    data, kind="overview", target="label"
)
fig.savefig("overview.png", dpi=150, bbox_inches="tight")

fig, ax = AutoDataMining.plot_preprocessing(
    data, kind="correlation", target="label"
)
fig.savefig("correlation.png", dpi=150, bbox_inches="tight")
~~~

Pilihan grafik meliputi ringkasan, heatmap nilai kosong, heatmap korelasi, distribusi fitur, boxplot untuk melihat outlier potensial, dan distribusi target. Fungsi mengembalikan figure dan axes; simpan dengan savefig atau tampilkan dengan show=True. Lihat [panduan prapemrosesan](docs/preprocessing.md) untuk semua nama kind dan keterangannya. Grafik memerlukan extra visualization.

## Explainability, laporan, dan visualisasi hasil

~~~python
# Dapatkan kesimpulan naratif cerdas dalam Bahasa Indonesia
print(results.summary())  # atau results.kesimpulan()

importance = results.explain(method="auto")
~~~

Fungsi `results.summary()` menghasilkan kesimpulan terstruktur yang siap dikutip untuk bab pembahasan skripsi/paper ilmiah, meliputi:
- Peringkat model terbaik dan alasan pemilihannya.
- Analisis generalisasi dan deteksi *overfitting* (komparasi skor data latih vs data uji).
- Interpretasi arti metrik (akurasi, presisi, recall, F1, R², MAE/RMSE, Silhouette, atau Lift) dalam kalimat bahasa manusia.
- Draf paragraf pembahasan siap pakai.

Mode otomatis pada `explain()` menggunakan feature importance bawaan model bila tersedia, atau permutation importance untuk tugas dengan label. SHAP memerlukan extra explainability. Feature importance menjelaskan perilaku model dan tidak membuktikan hubungan sebab-akibat. AutoDataMining tidak membuat feature importance untuk pengelompokan atau anomali tanpa dukungan model.

Beberapa grafik hasil yang tersedia:

~~~python
results.plot(kind="leaderboard")
results.plot(kind="confusion_matrix")          # Klasifikasi
results.plot(kind="roc_curve")                 # Klasifikasi biner
results.plot(kind="precision_recall_curve")    # Klasifikasi biner
results.plot(kind="residuals")                 # Regresi
results.plot(kind="clusters")                  # Pengelompokan
results.plot(kind="anomalies")                 # Deteksi anomali
results.plot(kind="rules")                     # Aturan asosiasi
~~~

Laporan HTML memuat metadata eksperimen, metrik, tabel peringkat model, dan batasan evaluasi. Baris dataset tidak dimasukkan ke laporan.

## Penyimpanan model

~~~python
results.save("model.joblib")
~~~

File yang disimpan berisi pipeline prapemrosesan dan model terpilih. Muat kembali dengan load_model, lalu gunakan predict pada data yang memiliki kolom fitur yang sesuai.

File joblib/pickle dapat menjalankan kode saat dimuat. Muat hanya file dari sumber tepercaya. Kompatibilitas antarversi scikit-learn tidak dijamin.

## Catatan evaluasi

- Klasifikasi dan regresi menggunakan pembagian data pelatihan dan data uji; validasi silang untuk memilih model berjalan pada bagian pelatihan.
- Prapemrosesan dilatih secara terpisah di setiap fold validasi silang.
- Stratifikasi digunakan untuk klasifikasi bila jumlah data tiap kelas memungkinkan.
- Skor validasi silang digunakan untuk memilih model jika tersedia. Metrik data uji dilaporkan terpisah dan sebaiknya tidak dipakai berulang kali untuk tuning.
- Pengelompokan dan deteksi anomali tanpa label acuan tidak memiliki metrik akurasi terhadap kelas sebenarnya.
- Pertimbangkan ketidakseimbangan kelas, kebocoran data, pembagian berdasarkan waktu/kelompok, kalibrasi, latensi, dan perubahan distribusi data.
- Hasil satu eksperimen tidak menjamin performa yang sama pada dataset atau penggunaan lain.

## Pemilihan estimator kustom

Estimator tambahan dapat didaftarkan tanpa mengubah source code library.

~~~python
from sklearn.linear_model import SGDClassifier
from autodatamining import AutoDataMining, EstimatorRegistry

registry = EstimatorRegistry()
registry.register(
    "fast_sgd",
    SGDClassifier,
    tasks="classification",
    params={"max_iter": 2000, "tol": 1e-3},
)

results = AutoDataMining(
    task="classification",
    target="label",
    registry=registry,
    models=["fast_sgd"],
    cv=3,
).fit(data)
~~~

Gunakan nama unik dan estimator yang kompatibel dengan task. Estimator akan di-clone sebelum dilatih. Parameter khusus model dapat diberikan melalui model_params.

## Seleksi fitur dan reduksi dimensi

~~~python
from autodatamining import select_features, reduce_dimensions

selected, selector = select_features(
    numeric_features,
    y,
    method="mutual_information",
    task="classification",
    k=5,
)
embedding, pca = reduce_dimensions(
    numeric_features,
    method="pca",
    n_components=2,
)
~~~

Fungsi ini melatih transformasi langsung pada data yang diberikan. Untuk evaluasi ilmiah, letakkan seleksi fitur atau reduksi dimensi dalam pipeline validasi silang agar setiap fold hanya mempelajari transformasi dari bagian pelatihan.

## Pengembangan dan pengujian

~~~bash
python -m pip install -e ".[dev,visualization]"
python -m compileall -q src tests
python -m ruff check src tests examples
python -m pytest -q
python -m build
python -m twine check dist/*
~~~

Lihat [panduan kontribusi](CONTRIBUTING.md), [changelog](CHANGELOG.md), [panduan preprocessing](docs/preprocessing.md), dan [panduan pemula](docs/panduan-penggunaan.md).

## PyPI

[AutoDataMining di PyPI](https://pypi.org/project/autodatamining)

## Lisensi

AutoDataMining menggunakan lisensi MIT. Lihat file [LICENSE](LICENSE).
