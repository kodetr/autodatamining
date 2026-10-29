"""Interactive educational glossary and concept explainer for data mining learners."""

from .exceptions import DataValidationError

_GLOSSARY = {
    "overfitting": {
        "title": "Overfitting (Terlalu Menghafal Data Latih)",
        "definisi": (
            "Kondisi di mana model machine learning mempelajari detail dan noise data latih terlalu sempurna, "
            "sehingga akurasi data latih sangat tinggi tetapi performanya anjlok saat menguji data baru."
        ),
        "analogi": (
            "Seperti siswa yang hanya menghafal kunci jawaban lembar soal tahun lalu tanpa memahami konsepnya. "
            "Saat ujian diberikan soal baru yang sedikit berbeda, nilainya langsung jatuh."
        ),
        "solusi": "Tambahkan jumlah data, gunakan Cross-Validation, kurangi kompleksitas model, atau gunakan regularisasi.",
    },
    "underfitting": {
        "title": "Underfitting (Model Terlalu Sederhana)",
        "definisi": (
            "Kondisi ketika model terlalu sederhana sehingga tidak mampu menangkap pola mendasar pada data. "
            "Baik pada data latih maupun data uji, performa model tetap buruk."
        ),
        "analogi": (
            "Seperti mencoba memprediksi harga rumah di seluruh kota hanya berdasarkan warna cat temboknya saja."
        ),
        "solusi": "Gunakan algoritma yang lebih ekspresif (misal: Random Forest atau Gradient Boosting) dan tambahkan fitur relevan.",
    },
    "cross_validation": {
        "title": "Cross-Validation (Validasi Silang K-Fold)",
        "definisi": (
            "Metode evaluasi statistik di mana data latih dibagi menjadi K lipatan (folds). Model dilatih K kali, "
            "bergantian menggunakan K-1 lipatan sebagai data latih dan 1 lipatan sebagai validasi."
        ),
        "analogi": (
            "Seperti latihan try-out ujian sebanyak 5 kali dengan paket soal yang berbeda-beda, lalu nilai rata-ratanya "
            "yang dijadikan tolok ukur kesiapan siswa."
        ),
        "solusi": "Selalu gunakan k-fold CV (biasanya 5 atau 10 fold) untuk memilih model terbaik secara objektif.",
    },
    "data_leakage": {
        "title": "Data Leakage (Kebocoran Data)",
        "definisi": (
            "Kesalahan fatal metodologi di mana informasi dari data uji secara tidak sengaja 'bocor' atau digunakan "
            "selama proses pelatihan (misalnya normalisasi atau imputasi dilakukan sebelum membagi train-test)."
        ),
        "analogi": (
            "Seperti seorang guru yang secara tidak sengaja memperlihatkan kisi-kisi soal ujian besok saat les pelajaran."
        ),
        "solusi": "Gunakan pipeline scikit-learn (seperti yang dilakukan AutoDataMining) agar prapemrosesan diisolasi pada data latih.",
    },
    "precision_recall": {
        "title": "Presisi vs Recall",
        "definisi": (
            "Presisi: Dari semua yang diprediksi positif, berapa persen yang benar-benar positif?\n"
            "Recall: Dari semua kasus positif di dunia nyata, berapa persen yang berhasil ditemukan model?"
        ),
        "analogi": (
            "Kasus deteksi penyakit: Presisi tinggi berarti jika dokter memvonis sakit, kemungkinan besar Anda memang sakit. "
            "Recall tinggi berarti dokter sangat teliti sehingga tidak ada pasien sakit yang terlewat pulang tanpa terdeteksi."
        ),
        "solusi": "Jika biaya salah vonis positif mahal gunakan Presisi. Jika bahaya lolos deteksi fatal gunakan Recall.",
    },
    "f1_score": {
        "title": "F1-Score (Keseimbangan Presisi dan Recall)",
        "definisi": (
            "Rata-rata harmonis antara Presisi dan Recall: 2 * (Presisi * Recall) / (Presisi + Recall). "
            "Sangat penting jika dataset memiliki kelas yang tidak seimbang (imbalanced data)."
        ),
        "analogi": (
            "Seperti mencari pemain sepak bola yang seimbang antara kemampuan menyerang dan bertahan."
        ),
        "solusi": "Gunakan F1-Score sebagai metrik utama jika target Anda tidak seimbang (misal: 95% data negatif, 5% positif).",
    },
    "lift": {
        "title": "Lift (Kekuatan Aturan Asosiasi)",
        "definisi": (
            "Rasio kemunculan bersama item A dan B dibandingkan jika keduanya muncul secara independen (kebetulan acak). "
            "Lift > 1 berarti ada keterkaitan positif yang kuat; Lift = 1 berarti independen; Lift < 1 berarti saling menolak."
        ),
        "analogi": (
            "Jika penjualan roti dan selai sama-sama melonjak di keranjang belanja yang sama melebihi peluang normal, Lift-nya > 1."
        ),
        "solusi": "Fokuskan analisis keranjang belanja pada aturan dengan nilai Lift > 1.0.",
    },
    "silhouette": {
        "title": "Skor Silhouette (Kualitas Klaster)",
        "definisi": (
            "Ukuran seberapa mirip suatu objek dengan kelompoknya sendiri dibandingkan kelompok lain. "
            "Rentang skor dari -1 sampai +1. Skor mendekati +1 berarti klaster padat dan terpisah tegas."
        ),
        "analogi": (
            "Seperti memisahkan murid ke dalam kelompok belajar. Jika anak-anak sangat akrab di kelompoknya dan tidak bingung pindah kelompok lain, skornya tinggi."
        ),
        "solusi": "Gunakan Silhouette > 0.5 sebagai target standar pengelompokan yang layak untuk laporan skripsi.",
    },
    "one_hot_encoding": {
        "title": "One-Hot Encoding (Konversi Kategori ke Biner)",
        "definisi": (
            "Teknik mengubah fitur teks/kategorikal menjadi kolom-kolom baru bernilai 0 atau 1, "
            "mencegah model menganggap kategori teks memiliki urutan hierarki numerik."
        ),
        "analogi": (
            "Mengisi formulir pilihan ganda: memberi centang pada satu kotak pilihan dan mengosongkan kotak lainnya."
        ),
        "solusi": "Wajib digunakan untuk data nominal seperti warna ('merah', 'biru') atau kota ('Jakarta', 'Surabaya').",
    },
    "ensemble": {
        "title": "Ensemble Learning (Voting & Stacking)",
        "definisi": (
            "Teknik menggabungkan prediksi dari beberapa model machine learning untuk menghasilkan prediksi gabungan "
            "yang lebih akurat dan lebih stabil daripada model tunggal mana pun."
        ),
        "analogi": (
            "Musyawarah dokter spesialis: Keputusan pengobatan diambil berdasarkan voting atau musyawarah banyak dokter ahli, bukan hanya opini satu dokter."
        ),
        "solusi": "Gunakan VotingClassifier atau StackingClassifier jika akurasi model tunggal sudah mencapai batas maksimal.",
    },
}


def list_concepts() -> list[str]:
    """Return all available conceptual topics in the glossary."""
    return sorted(_GLOSSARY.keys())


def explain_concept(concept: str, lang: str = "id") -> str:
    """Explain a data mining concept with plain-language definitions and everyday analogies.

    Parameters
    ----------
    concept : str
        Concept to explain (e.g. 'overfitting', 'cross_validation', 'lift', 'f1_score').
    lang : str, default='id'
        Language for explanation ('id').
    """
    key = str(concept).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cv": "cross_validation",
        "leakage": "data_leakage",
        "precision": "precision_recall",
        "recall": "precision_recall",
        "f1": "f1_score",
        "f_score": "f1_score",
        "ohe": "one_hot_encoding",
        "encoding": "one_hot_encoding",
        "stacking": "ensemble",
        "voting": "ensemble",
        "underfit": "underfitting",
        "overfit": "overfitting",
    }
    canonical = aliases.get(key, key)

    if canonical not in _GLOSSARY:
        available = ", ".join(list_concepts())
        raise DataValidationError(f"Concept {concept!r} not found. Available concepts: {available}")

    item = _GLOSSARY[canonical]
    output = (
        f"==================================================================\n"
        f"📚 KONSEP DATA MINING: {item['title'].upper()}\n"
        f"==================================================================\n"
        f"📖 Definisi:\n   {item['definisi']}\n\n"
        f"💡 Analogi Sehari-hari:\n   {item['analogi']}\n\n"
        f"🎯 Tips Praktis untuk Skripsi/Riset:\n   {item['solusi']}\n"
        f"=================================================================="
    )
    return output
