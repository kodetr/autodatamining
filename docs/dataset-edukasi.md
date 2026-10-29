# Katalog Dataset Edukasi Bawaan AutoDataMining

AutoDataMining menyediakan **5 dataset edukasi bawaan** (*built-in offline datasets*) yang dirancang khusus untuk mahasiswa, dosen, dan peneliti data mining. 

Dengan dataset bawaan ini:
- ❌ **Tidak perlu** mencari file CSV di internet.
- ❌ **Tidak perlu** koneksi internet (*100% offline & reproducible*).
- ❌ **Tidak perlu** pusing membersihkan format file yang rusak.
- ✅ Langsung siap pakai dalam 1 baris kode untuk praktikum, tugas akhir, atau studi banding algoritma.

---

## Daftar Fungsi Dataset

```python
from autodatamining import load_dataset, list_datasets

# Melihat daftar semua dataset yang tersedia
print(list_datasets())
# Output: ['kelulusan_mahasiswa', 'harga_rumah', 'segmentasi_nasabah', 'keranjang_belanja', 'deteksi_penipuan']

# Memuat salah satu dataset
data = load_dataset("kelulusan_mahasiswa")
```

---

## 1. Dataset Kelulusan Mahasiswa (`"kelulusan_mahasiswa"`)

* **Jenis Tugas:** Klasifikasi Biner (*Binary Classification*)
* **Jumlah Data:** 120 baris mahasiswa
* **Konteks Masalah:** Memprediksi apakah seorang mahasiswa akan lulus tepat waktu (8 semester) berdasarkan rekam jejak akademik semester 1–4 dan keaktifan organisasi.

### Struktur Kolom:
| Nama Kolom | Tipe Data | Deskripsi | Contoh Nilai |
| --- | --- | --- | --- |
| `ipk` | Float | Indeks Prestasi Kumulatif (IPK) s.d. semester 4 | `3.45`, `2.85` |
| `sks_semester_4` | Integer | Jumlah SKS yang telah lulus s.d. semester 4 | `84`, `72` |
| `kehadiran_persen` | Integer | Persentase kehadiran kuliah (%) | `90`, `75` |
| `ikut_organisasi` | String | Keikutsertaan dalam organisasi kampus | `"Ya"`, `"Tidak"` |
| **`lulus_tepat_waktu`** | String (Target) | Status kelulusan tepat waktu | `"Ya"`, `"Tidak"` |

### Contoh Penggunaan Cepat:
```python
from autodatamining import AutoDataMining, load_dataset

data = load_dataset("kelulusan_mahasiswa")
results = AutoDataMining(task="classification", target="lulus_tepat_waktu").fit(data)
print(results.leaderboard)
print(results.summary())
```

---

## 2. Dataset Harga Rumah (`"harga_rumah"`)

* **Jenis Tugas:** Regresi (*Regression / Continuous Estimation*)
* **Jumlah Data:** 100 baris properti rumah
* **Konteks Masalah:** Memperkirakan harga jual properti rumah tinggal berdasarkan karakteristik fisik bangunan dan lokasi geografisnya terhadap pusat kota.

### Struktur Kolom:
| Nama Kolom | Tipe Data | Deskripsi | Contoh Nilai |
| --- | --- | --- | --- |
| `luas_tanah` | Integer | Luas tanah dalam meter persegi (m²) | `120`, `250` |
| `luas_bangunan` | Integer | Luas bangunan dalam meter persegi (m²) | `90`, `180` |
| `kamar_tidur` | Integer | Jumlah kamar tidur | `3`, `4` |
| `jarak_pusat_kota_km` | Float | Jarak lokasi rumah ke pusat kota (kilometer) | `5.2`, `14.8` |
| **`harga_juta`** | Float (Target) | Estimasi harga properti (dalam Juta Rupiah) | `850.5`, `1450.0` |

### Contoh Penggunaan Cepat:
```python
from autodatamining import AutoDataMining, load_dataset

data = load_dataset("harga_rumah")
results = AutoDataMining(task="regression", target="harga_juta").fit(data)
print(results.leaderboard)
print(results.metrics)  # R², MAE, RMSE
```

---

## 3. Dataset Segmentasi Nasabah Bank (`"segmentasi_nasabah"`)

* **Jenis Tugas:** Pengelompokan / Segmentasi (*Clustering / Unsupervised*)
* **Jumlah Data:** 100 baris profil nasabah
* **Konteks Masalah:** Mengelompokkan nasabah perbankan ke dalam beberapa klaster alami berdasarkan kapasitas finansial dan perilaku belanja, guna menentukan strategi promosi kartu kredit dan investasi yang tepat sasaran.

### Struktur Kolom:
| Nama Kolom | Tipe Data | Deskripsi | Contoh Nilai |
| --- | --- | --- | --- |
| `usia` | Integer | Usia nasabah (tahun) | `28`, `52` |
| `pendapatan_tahunan_juta` | Float | Pendapatan kotor tahunan (Juta Rupiah) | `45.0`, `115.5` |
| `skor_pengeluaran_1_100` | Float | Indeks pengeluaran/konsumsi nasabah (skala 1–100) | `38.5`, `88.0` |
| `saldo_rekening_juta` | Float | Rata-rata saldo tabungan nasabah (Juta Rupiah) | `24.5`, `92.0` |

### Contoh Penggunaan Cepat:
```python
from autodatamining import AutoDataMining, load_dataset

data = load_dataset("segmentasi_nasabah")
results = AutoDataMining(task="clustering", models=["kmeans", "agglomerative"]).fit(data)
print(results.leaderboard)

# Analisis profil kelompok nasabah:
data["cluster"] = results.labels
print(data.groupby("cluster").mean().round(2))
```

---

## 4. Dataset Keranjang Belanja Transaksi (`"keranjang_belanja"`)

* **Jenis Tugas:** Penambangan Aturan Asosiasi (*Association Rule Mining / Market Basket Analysis*)
* **Jumlah Data:** 20 transaksi belanja minimarket/supermarket
* **Konteks Masalah:** Menemukan kombinasi produk yang paling sering dibeli bersamaan oleh konsumen (misalnya *kopi & gula*, *roti & mentega*) untuk merancang tata letak rak toko atau promo paket bundling.

### Format Data:
Dataset ini berbentuk `list of lists` berisi nama-nama produk dalam setiap struk belanja:
```python
[
    ["roti", "susu", "mentega"],
    ["roti", "mentega"],
    ["kopi", "gula"],
    ["kopi", "gula", "biskuit"],
    ...
]
```

### Contoh Penggunaan Cepat:
```python
from autodatamining import AutoDataMining, load_dataset

transaksi = load_dataset("keranjang_belanja")

# Menggunakan AutoDataMining
results = AutoDataMining(
    task="association",
    min_support=0.2,     # Minimal muncul di 20% transaksi
    min_confidence=0.5,  # Kepastian aturan minimal 50%
).fit(transaksi)

print("Aturan Asosiasi:")
print(results.rules[["antecedents", "consequents", "support", "confidence", "lift"]])
```

---

## 5. Dataset Deteksi Penipuan Transaksi (`"deteksi_penipuan"`)

* **Jenis Tugas:** Deteksi Anomali / Pencilan (*Anomaly Detection / Outlier Detection*)
* **Jumlah Data:** 100 baris transaksi keuangan digital
* **Konteks Masalah:** Mendeteksi transaksi keuangan mencurigakan (*fraud / card-skimming*) yang menyimpang dari kebiasaan normal, seperti penarikan nominal sangat besar di jam dini hari dan lokasi yang sangat jauh.

### Struktur Kolom:
| Nama Kolom | Tipe Data | Deskripsi | Contoh Nilai |
| --- | --- | --- | --- |
| `nominal_transaksi` | Float | Nilai transaksi dalam Rupiah (Rp) | `150,000` (normal) vs `25,000,000` (anomali) |
| `jarak_km` | Float | Jarak transaksi dari domisili pemilik (km) | `5.2` (normal) vs `850.0` (anomali) |
| `jam_transaksi` | Integer | Jam terjadinya transaksi (0–23) | `14` (normal) vs `03` (anomali) |
| `frekuensi_24jam` | Integer | Jumlah transaksi dalam kurun 24 jam terakhir | `2` (normal) vs `22` (anomali) |

### Contoh Penggunaan Cepat:
```python
from autodatamining import AutoDataMining, load_dataset

data = load_dataset("deteksi_penipuan")
results = AutoDataMining(task="anomaly", models=["isolation_forest", "lof"]).fit(data)

# Status: -1 = Anomali/Mencurigakan, 1 = Normal
data["status"] = ["Anomali" if l == -1 else "Normal" for l in results.labels]
print(data["status"].value_counts())
```
