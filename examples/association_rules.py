"""Contoh Aturan Asosiasi Nyata: Analisis Keranjang Belanja (Market Basket Analysis).

Menunjukkan penambangan pola pembelian pelanggan menggunakan Apriori & FP-Growth:
1. Memuat dataset 'keranjang_belanja'
2. Menjalankan penambangan frequent itemsets dengan Apriori dan FP-Growth
3. Menghasilkan aturan asosiasi (Support, Confidence, Lift)
4. Memfilter aturan dengan Lift > 1.2 untuk rekomendasi bundling produk
"""

from autodatamining import (
    TransactionEncoder,
    apriori,
    association_rules,
    fpgrowth,
    load_dataset,
)

# 1. Muat dataset transaksi belanja
transaksi = load_dataset("keranjang_belanja")
print(f"Total Transaksi: {len(transaksi)}")
print("Contoh 5 Transaksi Pertama:")
for i, t in enumerate(transaksi[:5], 1):
    print(f"  {i}. {t}")

# 2. Transformasi transaksi ke bentuk One-Hot Encoding
te = TransactionEncoder()
te_ary = te.fit(transaksi).transform(transaksi)
df = te.to_dataframe(te_ary)

# 3. Ekstraksi Frequent Itemsets dengan FP-Growth (min_support=0.2 = minimal muncul di 20% transaksi)
frequent_itemsets = fpgrowth(df, min_support=0.2, use_colnames=True)
print("\n=== Frequent Itemsets (Muncul >= 20% Transaksi) ===")
print(frequent_itemsets.sort_values(by="support", ascending=False).head(10))

# 4. Generate Aturan Asosiasi (min_threshold Confidence = 0.5)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("\n=== Aturan Asosiasi yang Ditemukan ===")
cols = ["antecedents", "consequents", "support", "confidence", "lift"]
print(rules[cols].sort_values(by="lift", ascending=False).head(10))

# 5. Rekomendasi Bisnis dari Aturan dengan Lift > 1.0 (Korelasi Positif Kuat)
strong_rules = rules[rules["lift"] > 1.2]
print(f"\nDitemukan {len(strong_rules)} aturan kuat dengan Lift > 1.2:")
for idx, r in strong_rules.iterrows():
    ant = list(r["antecedents"])
    con = list(r["consequents"])
    print(f"👉 Jika pelanggan membeli {ant}, maka kemungkinan besar ({r['confidence']*100:.1f}%) juga membeli {con} (Lift: {r['lift']:.2f})")
