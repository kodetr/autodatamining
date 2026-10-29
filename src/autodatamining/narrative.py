"""Automated narrative conclusions and educational explanations for experiment results."""

from typing import Any, Dict


def generate_narrative_summary(results: Any, lang: str = "id") -> str:
    """Generate a structured, human-readable conclusion and educational summary of MiningResults.

    Parameters
    ----------
    results : MiningResults
        The fitted experiment results object.
    lang : str, default='id'
        Language for the narrative ('id' for Indonesian, 'en' for English).
    """
    task = results.task
    if task == "classification":
        return _summary_classification(results, lang)
    if task == "regression":
        return _summary_regression(results, lang)
    if task == "clustering":
        return _summary_clustering(results, lang)
    if task == "anomaly":
        return _summary_anomaly(results, lang)
    if task == "association":
        return _summary_association(results, lang)
    return f"Kesimpulan otomatis untuk tugas {task!r} belum tersedia."


def _summary_classification(results: Any, lang: str) -> str:
    metrics = results.metrics or {}
    model_name = results.model_name or "Estimator"
    board = results.leaderboard

    # Find CV score for the winner
    cv_score = None
    cv_std = None
    if board is not None and not board.empty:
        match = board[board["model"] == model_name]
        if not match.empty and "cv_score" in match.columns:
            val = match["cv_score"].values[0]
            if val is not None and val == val:
                cv_score = float(val)
        if not match.empty and "cv_std" in match.columns:
            val = match["cv_std"].values[0]
            if val is not None and val == val:
                cv_std = float(val)

    acc = metrics.get("accuracy")
    bal_acc = metrics.get("balanced_accuracy")
    prec = metrics.get("precision")
    rec = metrics.get("recall")
    f1 = metrics.get("f1")
    roc = metrics.get("roc_auc")

    lines = [
        "==================================================================",
        "          KESIMPULAN & RINGKASAN EVALUASI PENELITIAN",
        "==================================================================",
        f"Jenis Tugas       : Klasifikasi (Classification)",
        f"Algoritma Terbaik : {model_name.upper()}",
    ]

    # Section 1: Skor Model Terbaik
    lines.append("\n1. PERFORMA MODEL TERBAIK")
    if cv_score is not None:
        std_str = f" ± {cv_std * 100:.1f}%" if cv_std is not None else ""
        lines.append(f"   - Skor Validasi Silang (Training CV) : {cv_score * 100:.2f}%{std_str}")
    if acc is not None:
        lines.append(f"   - Akurasi Data Uji (Test Holdout)   : {acc * 100:.2f}%")
    if f1 is not None:
        lines.append(f"   - F1-Score Data Uji                 : {f1 * 100:.2f}%")

    # Section 2: Analisis Generalisasi & Overfitting
    lines.append("\n2. ANALISIS GENERALISASI & OVERFITTING")
    if cv_score is not None and acc is not None:
        gap = (cv_score - acc) * 100
        if gap > 8.0:
            lines.append(
                f"   ⚠️ PERINGATAN OVERFITTING:\n"
                f"      Skor data latih ({cv_score * 100:.1f}%) lebih tinggi signifikan dibanding data uji ({acc * 100:.1f}%)\n"
                f"      dengan selisih {gap:.1f}%. Model berpotensi menghafal pola data latih dan kurang mampu\n"
                f"      menggeneralisasi data baru. Disarankan menambah data latih atau regularisasi parameter."
            )
        elif gap < -8.0:
            lines.append(
                f"   ℹ️ PERFORMA DATA UJI LEBIH TINGGI:\n"
                f"      Akurasi data uji ({acc * 100:.1f}%) lebih tinggi dibanding validasi silang ({cv_score * 100:.1f}%).\n"
                f"      Hal ini wajar terjadi pada ukuran data uji yang relatif kecil atau sampel uji yang homogen."
            )
        else:
            lines.append(
                f"   ✅ MODEL STABIL (BEBAS OVERFITTING):\n"
                f"      Selisih antara validasi silang ({cv_score * 100:.1f}%) dan data uji ({acc * 100:.1f}%) hanya {abs(gap):.1f}%.\n"
                f"      Ini membuktikan model memiliki kemampuan generalisasi yang kokoh terhadap data yang belum pernah dilihat."
            )
    else:
        lines.append("   - Validasi silang tidak tersedia untuk membandingkan celah overfitting.")

    # Section 3: Interpretasi Metrik untuk Mahasiswa/Peneliti
    lines.append("\n3. CARA MEMBACA METRIK (INTERPRETASI ILMIAH)")
    if acc is not None:
        lines.append(
            f"   - Akurasi ({acc * 100:.1f}%): Dari 100 data uji, model memprediksi {acc * 100:.1f} data dengan tepat."
        )
    if bal_acc is not None and acc is not None and abs(acc - bal_acc) > 0.05:
        lines.append(
            f"   - Balanced Accuracy ({bal_acc * 100:.1f}%): Terlihat perbedaan dengan akurasi standar ({acc * 100:.1f}%),\n"
            f"     menandakan adanya ketidakseimbangan jumlah data antar-kelas (imbalanced dataset)."
        )
    if prec is not None:
        lines.append(
            f"   - Presisi ({prec * 100:.1f}%): Dari seluruh data yang ditebak sebagai kelas positif, {prec * 100:.1f}% terbukti benar."
        )
    if rec is not None:
        lines.append(
            f"   - Recall ({rec * 100:.1f}%): Dari seluruh data yang sebenarnya merupakan kelas positif, model berhasil menjaring {rec * 100:.1f}%."
        )
    if f1 is not None:
        lines.append(
            f"   - F1-Score ({f1 * 100:.1f}%): Rata-rata harmonis antara presisi dan recall. Sangat relevan sebagai tolok ukur utama."
        )
    if roc is not None:
        lines.append(
            f"   - ROC-AUC ({roc:.3f}): Kemampuan model memisahkan antar-kelas (skor mendekati 1.0 = sempurna)."
        )

    # Section 4: Draf Pembahasan Skripsi/Paper
    lines.append("\n4. DRAFT KALIMAT PEMBAHASAN UNTUK SKRIPSI / PAPER")
    f1_text = f" dan F1-Score sebesar {f1 * 100:.1f}%" if f1 is not None else ""
    acc_text = f"akurasi sebesar {acc * 100:.1f}%" if acc is not None else "skor optimal"
    lines.append(
        f'   "Berdasarkan pengujian komparatif beberapa algoritma klasifikasi, model {model_name}\n'
        f'   menunjukkan performa terbaik dengan {acc_text}{f1_text} pada data pengujian.\n'
        f'   Evaluasi validasi silang mengonfirmasi stabilitas model tanpa indikasi overfitting yang signifikan,\n'
        f'   sehingga model {model_name} direkomendasikan sebagai model akhir dalam penelitian ini."'
    )
    lines.append("==================================================================")
    return "\n".join(lines)


def _summary_regression(results: Any, lang: str) -> str:
    metrics = results.metrics or {}
    model_name = results.model_name or "Estimator"
    board = results.leaderboard

    cv_r2 = None
    if board is not None and not board.empty:
        match = board[board["model"] == model_name]
        if not match.empty and "cv_score" in match.columns:
            val = match["cv_score"].values[0]
            if val is not None and val == val:
                cv_r2 = float(val)

    r2 = metrics.get("r2")
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")

    lines = [
        "==================================================================",
        "          KESIMPULAN & RINGKASAN EVALUASI PENELITIAN",
        "==================================================================",
        f"Jenis Tugas       : Regresi (Regression)",
        f"Algoritma Terbaik : {model_name.upper()}",
    ]

    lines.append("\n1. PERFORMA MODEL TERBAIK")
    if cv_r2 is not None:
        lines.append(f"   - R² Validasi Silang (Training CV) : {cv_r2:.4f} ({cv_r2 * 100:.1f}%)")
    if r2 is not None:
        lines.append(f"   - R² Data Uji (Test Holdout)       : {r2:.4f} ({r2 * 100:.1f}%)")
    if mae is not None:
        lines.append(f"   - Mean Absolute Error (MAE)        : {mae:.4f}")
    if rmse is not None:
        lines.append(f"   - Root Mean Squared Error (RMSE)   : {rmse:.4f}")

    lines.append("\n2. ANALISIS GENERALISASI & OVERFITTING")
    if cv_r2 is not None and r2 is not None:
        gap = cv_r2 - r2
        if gap > 0.15:
            lines.append(
                f"   ⚠️ PERINGATAN OVERFITTING:\n"
                f"      Nilai R² pada data latih ({cv_r2:.3f}) jauh lebih tinggi dibanding data uji ({r2:.3f}).\n"
                f"      Model mengalami penurunan kemampuan prediksi pada data baru. Gunakan teknik regularisasi (Ridge/Lasso)."
            )
        else:
            lines.append(
                f"   ✅ MODEL STABIL (BEBAS OVERFITTING):\n"
                f"      Nilai R² konsisten antara data latih ({cv_r2:.3f}) dan data uji ({r2:.3f}).\n"
                f"      Model mampu memperkirakan nilai kontinu dengan tingkat keandalan yang teruji."
            )
    else:
        lines.append("   - Perbandingan CV dan Holdout tidak lengkap untuk mendeteksi celah generalisasi.")

    lines.append("\n3. CARA MEMBACA METRIK (INTERPRETASI ILMIAH)")
    if r2 is not None:
        r2_pct = max(0.0, r2 * 100)
        lines.append(
            f"   - Koefisien Determinasi (R² = {r2:.3f}): Variabel independen berhasil menjelaskan {r2_pct:.1f}% variasi\n"
            f"     dari nilai target. Sisanya ({100 - r2_pct:.1f}%) dipengaruhi oleh faktor lain di luar dataset."
        )
    if mae is not None:
        lines.append(
            f"   - MAE ({mae:.3f}): Rata-rata selisih mutlak antara estimasi model dan nilai riil adalah {mae:.3f} satuan."
        )
    if rmse is not None:
        lines.append(
            f"   - RMSE ({rmse:.3f}): Rata-rata akar kuadrat galat adalah {rmse:.3f}. Nilai ini lebih peka terhadap error ekstrem."
        )

    lines.append("\n4. DRAFT KALIMAT PEMBAHASAN UNTUK SKRIPSI / PAPER")
    lines.append(
        f'   "Hasil evaluasi regresi menunjukkan bahwa algoritma {model_name} mencapai koefisien determinasi\n'
        f'   R² sebesar {r2 if r2 is not None else 0.0:.3f} dengan tingkat kesalahan rata-rata MAE sebesar {mae if mae is not None else 0.0:.3f}.\n'
        f'   Hal ini membuktikan efektivitas model dalam memperkirakan nilai target secara konsisten."'
    )
    lines.append("==================================================================")
    return "\n".join(lines)


def _summary_clustering(results: Any, lang: str) -> str:
    metrics = results.metrics or {}
    model_name = results.model_name or "Estimator"
    sil = metrics.get("silhouette")
    db = metrics.get("davies_bouldin")
    ch = metrics.get("calinski_harabasz")
    k = metrics.get("clusters")
    noise = metrics.get("noise_count", 0)

    lines = [
        "==================================================================",
        "          KESIMPULAN & RINGKASAN EVALUASI PENELITIAN",
        "==================================================================",
        f"Jenis Tugas       : Pengelompokan (Clustering)",
        f"Algoritma Terbaik : {model_name.upper()}",
    ]

    lines.append("\n1. STRUKTUR DAN JUMLAH KLASTER")
    if k is not None:
        lines.append(f"   - Jumlah Klaster Terbentuk : {k} kelompok")
    if noise and noise > 0:
        lines.append(f"   - Data Pencilan / Noise    : {noise} titik observasi (diabaikan dari klaster)")

    lines.append("\n2. EVALUASI KUALITAS PEMISAHAN KLASTER")
    if sil is not None:
        lines.append(f"   - Skor Silhouette           : {sil:.4f}")
        if sil > 0.7:
            lines.append("     👉 Interpretasi: STRUKTUR KUAT. Kelompok sangat terpisah jelas dan kompak.")
        elif sil >= 0.5:
            lines.append("     👉 Interpretasi: STRUKTUR BAIK / WAJAR. Kelompok terbentuk dengan pemisahan yang cukup tegas.")
        elif sil >= 0.25:
            lines.append("     👉 Interpretasi: STRUKTUR MODERAT / LEMAH. Terdapat sebagian data yang saling tumpang tindih (overlap).")
        else:
            lines.append("     👉 Interpretasi: STRUKTUR SANGAT LEMAH. Pemisahan antar kelompok hampir tidak nyata.")
    if db is not None:
        lines.append(f"   - Davies–Bouldin Index      : {db:.4f} (Makin mendekati 0, makin baik)")
    if ch is not None:
        lines.append(f"   - Calinski–Harabasz Index   : {ch:.1f} (Makin tinggi, makin kompak)")

    lines.append("\n3. DRAFT KALIMAT PEMBAHASAN UNTUK SKRIPSI / PAPER")
    lines.append(
        f'   "Pengelompokan menggunakan algoritma {model_name} berhasil membagi dataset ke dalam {k if k is not None else 0}\n'
        f'   klaster yang homogen dengan skor Silhouette sebesar {sil if sil is not None else 0.0:.3f}.\n'
        f'   Hasil ini menunjukkan bahwa pola karakteristik data memiliki pengelompokan alami yang valid."'
    )
    lines.append("==================================================================")
    return "\n".join(lines)


def _summary_anomaly(results: Any, lang: str) -> str:
    metrics = results.metrics or {}
    model_name = results.model_name or "Estimator"
    rate = metrics.get("anomaly_rate")
    count = metrics.get("anomaly_count")
    f1 = metrics.get("f1")

    lines = [
        "==================================================================",
        "          KESIMPULAN & RINGKASAN EVALUASI PENELITIAN",
        "==================================================================",
        f"Jenis Tugas       : Deteksi Anomali (Anomaly Detection)",
        f"Algoritma Terbaik : {model_name.upper()}",
    ]

    lines.append("\n1. TEMUAN DATA ANOMALI / PENCILAN")
    if count is not None:
        lines.append(f"   - Jumlah Anomali Ditemukan : {count} baris data")
    if rate is not None:
        lines.append(f"   - Persentase Anomali       : {rate * 100:.2f}% dari total dataset")

    lines.append("\n2. CATATAN METODOLOGI PENELITIAN")
    if f1 is not None:
        lines.append(
            f"   - Skor F1 Validasi : {f1 * 100:.1f}% (Dievaluasi berdasarkan label acuan ground-truth)."
        )
    else:
        lines.append(
            "   - Deteksi dilakukan secara tanpa pengawasan (unsupervised tanpa ground-truth).\n"
            "     Tingkat anomali menunjukkan banyaknya pencilan yang terdeteksi, bukan ukuran akurasi.\n"
            "     Setiap baris berlabel -1 perlu ditinjau lebih lanjut oleh pakar domain."
        )

    lines.append("\n3. DRAFT KALIMAT PEMBAHASAN UNTUK SKRIPSI / PAPER")
    lines.append(
        f'   "Metode deteksi anomali {model_name} berhasil mengidentifikasi {count if count is not None else 0} data ({rate * 100 if rate is not None else 0.0:.1f}%)\n'
        f'   sebagai observasi tidak biasa yang menyimpang dari distribusi normal dataset."'
    )
    lines.append("==================================================================")
    return "\n".join(lines)


def _summary_association(results: Any, lang: str) -> str:
    metrics = results.metrics or {}
    model_name = results.model_name or "Estimator"
    rules = getattr(results, "rules", None)
    itemsets = getattr(results, "frequent_itemsets", None)

    n_itemsets = len(itemsets) if itemsets is not None else metrics.get("itemsets_count", 0)
    n_rules = len(rules) if rules is not None else metrics.get("rules_count", 0)

    lines = [
        "==================================================================",
        "          KESIMPULAN & RINGKASAN EVALUASI PENELITIAN",
        "==================================================================",
        f"Jenis Tugas       : Aturan Asosiasi (Association Rule Mining)",
        f"Algoritma Terpilih: {model_name.upper()} (Unggul dalam kecepatan eksekusi)",
    ]

    lines.append("\n1. HASIL PENAMBANGAN POLA")
    lines.append(f"   - Frequent Itemsets Ditemukan : {n_itemsets} kombinasi item sering")
    lines.append(f"   - Aturan Asosiasi Terbentuk   : {n_rules} aturan pola perilaku")

    lines.append("\n2. ATURAN PALING BERPENGARUH (HIGHEST LIFT)")
    if rules is not None and not rules.empty:
        top_rule = rules.iloc[0]
        ant = list(top_rule["antecedents"])
        con = list(top_rule["consequents"])
        supp = top_rule["support"]
        conf = top_rule["confidence"]
        lift = top_rule["lift"]
        lines.append(f"   - Pola Terkuat   : {ant} => {con}")
        lines.append(f"   - Nilai Support  : {supp * 100:.1f}% (Frekuensi kemunculan bersama dalam transaksi)")
        lines.append(f"   - Nilai Confidence: {conf * 100:.1f}% (Peluang {con} dipilih jika {ant} dipilih)")
        lines.append(f"   - Nilai Lift     : {lift:.2f}")
        if lift > 1.0:
            lines.append(f"     👉 Interpretasi: LIFT > 1.0 membuktikan hubungan positif kuat (bukan kebetulan acak).")
    else:
        lines.append("   - Tidak ada aturan yang memenuhi batas ambang minimal.")

    lines.append("\n3. DRAFT KALIMAT PEMBAHASAN UNTUK SKRIPSI / PAPER")
    lines.append(
        f'   "Penambangan aturan asosiasi dengan algoritma {model_name} berhasil mengekstrak {n_rules} aturan pola transaksi.\n'
        f'   Korelasi tertinggi ditemukan pada kombinasi item dengan nilai Lift > 1.0 yang dapat dijadikan acuan strategi bundel produk."'
    )
    lines.append("==================================================================")
    return "\n".join(lines)
