# Panduan Penggunaan Aturan Asosiasi (Association Rules)

**Aturan Asosiasi (Association Rule Mining)** adalah teknik data mining tanpa pengawasan (*unsupervised*) yang digunakan untuk menemukan hubungan, pola, atau korelasi yang menarik antar-item dalam kumpulan data transaksi berukuran besar.

Teknik ini sangat populer untuk:
- **Market Basket Analysis (Analisis Keranjang Belanja):** Mengetahui produk apa yang sering dibeli bersamaan (misal: "Jika membeli roti, maka 80% kemungkinan juga membeli selai").
- **Analisis Pola Perilaku Pengguna / Web Usage Mining.**
- **Diagnosa Medis dan Gejala Klinis.**

AutoDataMining menyediakan dua algoritma utama:
1. **Apriori:** Algoritma klasik dengan pendekatan *level-wise join and prune*.
2. **FP-Growth (Frequent Pattern Growth):** Algoritma modern berbasis struktur pohon (*FP-Tree*) yang jauh lebih cepat dan efisien tanpa membangkitkan kandidat secara berulang.

---

## 1. Konsep dan Metrik Utama

Setiap aturan asosiasi berbentuk:
$$\text{Antecedent } (A) \rightarrow \text{Consequent } (C)$$
*(Contoh: $\{\text{Roti}\} \rightarrow \{\text{Selai}\})$*

| Metrik | Rumus | Arti Metrik |
| :--- | :--- | :--- |
| **Support** | $\frac{\text{Jumlah Transaksi } (A \cup C)}{\text{Total Transaksi } (N)}$ | Seberapa sering item $A$ dan $C$ muncul bersama dalam dataset. |
| **Confidence** | $\frac{\text{Support}(A \cup C)}{\text{Support}(A)}$ | Kepastian bahwa jika pelanggan membeli $A$, mereka juga membeli $C$. |
| **Lift** | $\frac{\text{Confidence}(A \to C)}{\text{Support}(C)}$ | Mengukur kekuatan aturan dibandingkan kemunculan acak. Nilai **$> 1$** menandakan korelasi positif yang kuat. |
| **Leverage** | $\text{Support}(A \cup C) - (\text{Support}(A) \times \text{Support}(C))$ | Selisih frekuensi observasi nyata dengan independensi teoritis. |
| **Conviction** | $\frac{1 - \text{Support}(C)}{1 - \text{Confidence}(A \to C)}$ | Rasio kegagalan aturan jika $A$ dan $C$ sepenuhnya independen. |

---

## 2. Cara Penggunaan Sederhana via AutoDataMining

Anda dapat langsung memasukkan daftar transaksi mentah (daftar item per transaksi):

```python
from autodatamining import AutoDataMining

# Data transaksi belanjaan
transaksi = [
    ["roti", "susu", "mentega"],
    ["roti", "mentega"],
    ["susu", "roti", "keju"],
    ["biskuit", "kopi"],
    ["roti", "mentega", "keju"],
    ["susu", "kopi", "biskuit"],
]

# Jalankan penambangan aturan asosiasi
results = AutoDataMining(
    task="association",
    min_support=0.3,       # Minimal muncul di 30% transaksi
    min_confidence=0.6,    # Minimal tingkat keyakinan 60%
    metric="confidence",   # Metrik penyaringan aturan
).fit(transaksi)

# Tampilkan perbandingan kecepatan algoritma
print("--- Leaderboard Algoritma ---")
print(results.leaderboard)

# Tampilkan kumpulan itemset yang sering muncul
print("\n--- Frequent Itemsets ---")
print(results.frequent_itemsets)

# Tampilkan aturan asosiasi yang dihasilkan
print("\n--- Aturan Asosiasi ---")
print(results.rules[["antecedents", "consequents", "support", "confidence", "lift"]])
```

---

## 3. Visualisasi Aturan Asosiasi

AutoDataMining dapat menampilkan sebaran aturan asosiasi dalam bentuk grafik hubungan *Support* vs *Confidence* dengan warna gradasi menunjukkan *Lift*:

```python
# Tampilkan grafik aturan (memerlukan matplotlib)
results.plot(kind="rules", show=True)
```

---

## 4. Penggunaan Fungsi Standalone (Apriori & FP-Growth)

Bagi peneliti atau mahasiswa yang ingin kontrol langsung pada data matriks tabel (one-hot binary DataFrame):

```python
import pandas as pd
from autodatamining import TransactionEncoder, apriori, fpgrowth, association_rules

transaksi = [
    ["apel", "pisang"],
    ["pisang", "jeruk", "mangga"],
    ["apel", "pisang", "jeruk"],
    ["apel", "jeruk"],
]

# 1. Ubah daftar transaksi menjadi DataFrame biner
te = TransactionEncoder()
df_biner = te.fit_transform(transaksi)
print(df_biner)

# 2. Temukan frequent itemsets dengan FP-Growth atau Apriori
frequent_patterns = fpgrowth(df_biner, min_support=0.5, use_colnames=True)
print(frequent_patterns)

# 3. Hasilkan aturan asosiasi dengan batas minimal Lift > 1.2
rules = association_rules(frequent_patterns, metric="lift", min_threshold=1.2)
print(rules)
```

---

## 5. Ringkasan Rekomendasi untuk Peneliti & Mahasiswa

- Gunakan **FP-Growth** untuk dataset besar atau dengan banyak variasi item karena performanya jauh lebih cepat dibanding Apriori.
- Nilai **Lift $> 1$** menunjukkan aturan yang bermakna secara praktis di lapangan (bukan sekadar kebetulan).
- Hasil aturan dapat diekspor ke file CSV via `results.rules.to_csv("aturan_asosiasi.csv", index=False)`.
