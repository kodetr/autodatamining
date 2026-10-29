"""AutoDataMining - No-Code Interactive Web Studio.

Antarmuka web grafis (GUI) no-code untuk eksperimen data mining, perbandingan model otomatis,
visualisasi alur pipeline, penyetelan hyperparameter, dan ekspor publikasi ilmiah / skripsi.
Siap dijalankan di Hugging Face Spaces atau secara lokal.
"""

import io
import os
import sys
import tempfile
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Pastikan module autodatamining dapat diimpor baik dari instalasi pip maupun source lokal
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from autodatamining import (
    AutoDataMining,
    TransactionEncoder,
    apriori,
    association_rules,
    explain_concept,
    fpgrowth,
    list_concepts,
    list_datasets,
    load_dataset,
)

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="AutoDataMining Studio - No-Code GUI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk estetika modern, elegan, dan profesional
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #e0f2fe;
        margin: 6px 0 0 0;
        font-size: 1.05rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Utama
st.markdown(
    """
    <div class="main-header">
        <h1>🚀 AutoDataMining Studio</h1>
        <p>Platform No-Code Interaktif untuk Pembelajaran Data Mining, Riset, dan Penyusunan Skripsi</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Inisialisasi Session State
if "results" not in st.session_state:
    st.session_state.results = None
if "df" not in st.session_state:
    st.session_state.df = None
if "tuned_results" not in st.session_state:
    st.session_state.tuned_results = None
if "task_type" not in st.session_state:
    st.session_state.task_type = "classification"

# Sidebar: Sumber Data & Pengaturan
with st.sidebar:
    st.header("⚙️ Konfigurasi Data & Tugas")
    
    data_source = st.radio(
        "Pilih Sumber Dataset:",
        ("Dataset Edukasi Bawaan (1-Klik)", "Unggah File CSV / Excel Sendiri"),
    )
    
    raw_data = None
    task_hint = "classification"
    
    if data_source == "Dataset Edukasi Bawaan (1-Klik)":
        dataset_choice = st.selectbox(
            "Pilih Dataset Edukasi:",
            (
                "kelulusan_mahasiswa (Klasifikasi)",
                "harga_rumah (Regresi)",
                "segmentasi_nasabah (Clustering)",
                "keranjang_belanja (Aturan Asosiasi)",
                "deteksi_penipuan (Deteksi Anomali)",
            ),
        )
        ds_key = dataset_choice.split(" ")[0]
        if ds_key == "keranjang_belanja":
            raw_data = load_dataset("keranjang_belanja")
            task_hint = "association"
        else:
            raw_data = load_dataset(ds_key)
            if ds_key == "kelulusan_mahasiswa":
                task_hint = "classification"
            elif ds_key == "harga_rumah":
                task_hint = "regression"
            elif ds_key == "segmentasi_nasabah":
                task_hint = "clustering"
            elif ds_key == "deteksi_penipuan":
                task_hint = "anomaly"
        st.success(f"Dataset bawaan `{ds_key}` siap digunakan!")
    else:
        uploaded_file = st.file_uploader(
            "Unggah file data (CSV atau Excel):",
            type=["csv", "xlsx", "xls"],
            help="Maksimal ukuran file 20MB. Format CSV atau XLSX.",
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    raw_data = pd.read_csv(uploaded_file)
                else:
                    raw_data = pd.read_excel(uploaded_file)
                st.success(f"Berhasil membaca `{uploaded_file.name}` ({len(raw_data)} baris)")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    st.markdown("---")
    st.subheader("🎯 Pemilihan Tugas Data Mining")
    
    task_options = ["classification", "regression", "clustering", "association", "anomaly"]
    task_labels = {
        "classification": "Klasifikasi (Classification)",
        "regression": "Regresi (Regression)",
        "clustering": "Pengelompokan (Clustering)",
        "association": "Aturan Asosiasi (Market Basket)",
        "anomaly": "Deteksi Anomali (Outlier/Fraud)",
    }
    
    default_task_idx = task_options.index(task_hint) if task_hint in task_options else 0
    selected_task = st.selectbox(
        "Pilar Data Mining:",
        task_options,
        index=default_task_idx,
        format_func=lambda x: task_labels[x],
    )
    st.session_state.task_type = selected_task

    target_col = None
    cv_folds = 5
    test_ratio = 0.2
    min_sup = 0.2
    min_conf = 0.5

    if isinstance(raw_data, pd.DataFrame) and selected_task in {"classification", "regression"}:
        cols = list(raw_data.columns)
        default_target_idx = len(cols) - 1
        if "lulus_tepat_waktu" in cols:
            default_target_idx = cols.index("lulus_tepat_waktu")
        elif "harga_juta" in cols:
            default_target_idx = cols.index("harga_juta")
            
        target_col = st.selectbox("Kolom Target (Yang Ingin Diprediksi):", cols, index=default_target_idx)
        test_ratio = st.slider("Ukuran Data Uji (Holdout Ratio):", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Lipatan Validasi Silang (CV Folds):", 2, 10, 5, 1)
    elif selected_task == "association":
        min_sup = st.slider("Minimum Support (Frekuensi Kemunculan):", 0.05, 0.8, 0.2, 0.05)
        min_conf = st.slider("Minimum Confidence (Tingkat Kepastian):", 0.1, 1.0, 0.5, 0.05)

    st.markdown("---")
    run_btn = st.button("🚀 Jalankan Penambangan Data Sekarang", type="primary", use_container_width=True)

# Simpan dataset ke session state jika baru
if raw_data is not None:
    st.session_state.df = raw_data

# Eksekusi Pemodelan jika tombol ditekan
if run_btn:
    if st.session_state.df is None:
        st.warning("Silakan pilih dataset bawaan atau unggah file CSV terlebih dahulu!")
    else:
        with st.spinner("⏳ Menjalankan algoritma data mining dan perbandingan model otomatis..."):
            try:
                data_input = st.session_state.df
                st.session_state.tuned_results = None  # Reset tuning sebelumnya
                
                if selected_task == "association":
                    results = AutoDataMining(
                        task="association",
                        min_support=min_sup,
                        min_confidence=min_conf,
                    ).fit(data_input)
                elif selected_task in {"classification", "regression"}:
                    results = AutoDataMining(
                        task=selected_task,
                        target=target_col,
                        test_size=test_ratio,
                        cv=cv_folds,
                        random_state=42,
                    ).fit(data_input)
                elif selected_task == "clustering":
                    results = AutoDataMining(
                        task="clustering",
                        random_state=42,
                    ).fit(data_input)
                elif selected_task == "anomaly":
                    results = AutoDataMining(
                        task="anomaly",
                        random_state=42,
                    ).fit(data_input)
                    
                st.session_state.results = results
                st.success("🎉 Penambangan data selesai! Lihat hasil lengkap pada tab di bawah.")
            except Exception as exc:
                st.error(f"Terjadi kesalahan saat pemrosesan: {exc}")
                st.code(traceback.format_exc())

# Tampilan Konten Utama Berbasis Tab
tab_data, tab_leaderboard, tab_viz, tab_summary, tab_tuning, tab_export, tab_predict = st.tabs([
    "📊 1. Eksplorasi Data",
    "🏆 2. Peringkat Model",
    "📈 3. Visualisasi Grafik",
    "📝 4. Kesimpulan Ilmiah",
    "⚡ 5. Auto Tuning",
    "🎓 6. Ekspor Skripsi / LaTeX",
    "🔮 7. Uji Prediksi Langsung",
])

# ----------------- TAB 1: EKSPLORASI DATA -----------------
with tab_data:
    st.subheader("Pratinjau & Karakteristik Data")
    if st.session_state.df is not None:
        df = st.session_state.df
        if isinstance(df, pd.DataFrame):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Baris (Observasi)", f"{len(df):,}")
            with col_b:
                st.metric("Total Kolom (Fitur)", f"{df.shape[1]:,}")
            with col_c:
                st.metric("Total Nilai Kosong (NaN)", f"{df.isna().sum().sum():,}")
                
            st.markdown("##### 10 Baris Pertama Data:")
            st.dataframe(df.head(10), use_container_width=True)
            
            with st.expander("🔍 Informasi Tipe Data & Statistik Deskriptif"):
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write("**Tipe Kolom:**")
                    st.write(df.dtypes.astype(str))
                with col_d2:
                    st.write("**Statistik Ringkas:**")
                    st.write(df.describe())
        elif isinstance(df, (list, tuple)):
            st.info(f"Dataset Transaksi Keranjang Belanja ({len(df)} transaksi).")
            st.write("**5 Transaksi Pertama:**")
            for idx, trans in enumerate(df[:5], 1):
                st.write(f"- Transaksi #{idx}: `{trans}`")
    else:
        st.info("Pilih dataset di bilah sisi kiri untuk mulai bereksplorasi.")

# ----------------- TAB 2: LEADERBOARD -----------------
with tab_leaderboard:
    st.subheader("Tabel Peringkat Model (Leaderboard)")
    results = st.session_state.results
    if results is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            best_name = getattr(results, "model_name", "N/A")
            st.metric("Algoritma Terbaik", str(best_name).upper())
        with c2:
            if results.task in {"classification", "regression"}:
                top_cv = results.leaderboard.iloc[0]["cv_score"]
                st.metric("Skor Validasi Silang (CV)", f"{top_cv:.4f}")
            elif results.task == "clustering":
                sil = results.leaderboard.iloc[0].get("silhouette", 0.0)
                st.metric("Skor Silhouette", f"{sil:.4f}")
            elif results.task == "association":
                n_rules = len(results.rules) if results.rules is not None else 0
                st.metric("Aturan Ditemukan", f"{n_rules} aturan")
            elif results.task == "anomaly":
                anom_rate = results.leaderboard.iloc[0].get("anomaly_rate", 0.0)
                st.metric("Persentase Anomali", f"{anom_rate*100:.1f}%")
        with c3:
            train_sec = getattr(results, "metrics", {}).get("training_seconds", 0.0)
            st.metric("Waktu Komparasi", f"{train_sec:.3f} detik")

        st.markdown("##### Tabel Perbandingan Kinerja Semua Algoritma:")
        st.dataframe(results.leaderboard, use_container_width=True)
        
        csv_data = results.leaderboard.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Unduh Tabel Leaderboard (.csv)",
            csv_data,
            f"leaderboard_{results.task}.csv",
            "text/csv",
        )
    else:
        st.info("Tekan tombol **'🚀 Jalankan Penambangan Data Sekarang'** pada bilah sisi untuk melihat perbandingan model.")

# ----------------- TAB 3: VISUALISASI GRAFIK -----------------
with tab_viz:
    st.subheader("📈 Visualisasi Diagnostik & Arsitektur Pipeline")
    results = st.session_state.results
    if results is not None:
        # 1. Pilihan Model untuk Evaluasi (Jika Classification / Regression)
        if results.task in {"classification", "regression"}:
            model_options = list(results.leaderboard["model"].unique())
            selected_model = st.selectbox(
                "🎯 Pilih Algoritma yang Ingin Dilihat Evaluasi Grafiknya:",
                model_options,
                index=0,
                format_func=lambda x: f"{x.upper()} {'(🏆 Pemenang Leaderboard)' if x == results.model_name else ''}",
            )
        else:
            selected_model = results.model_name

        st.markdown("---")
        
        # 2. Grafik Diagnostik Spesifik Sesuai Pilar Tugas
        if results.task == "classification":
            st.markdown(f"#### 🎯 Evaluasi Model: `{selected_model.upper()}`")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.write("**Matriks Kebingungan (Confusion Matrix):**")
                fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))
                try:
                    results.plot(kind="confusion_matrix", ax=ax_cm, model_name=selected_model)
                    st.pyplot(fig_cm)
                except Exception as e:
                    st.warning(f"Tidak dapat menggambar matriks kebingungan: {e}")
                finally:
                    plt.close(fig_cm)
                st.info("💡 **Cara Membaca:** Kotak diagonal (kiri-atas dan kanan-bawah) menunjukkan prediksi yang **BENAR**. Kotak di luar diagonal menunjukkan kesalahan prediksi (*False Positives / False Negatives*).")

            with col_v2:
                st.write("**Kurva ROC (Receiver Operating Characteristic):**")
                fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4.2))
                try:
                    results.plot(kind="roc_curve", ax=ax_roc, model_name=selected_model)
                    st.pyplot(fig_roc)
                except Exception:
                    # Fallback jika multi-class: tampilkan grafik batang leaderboard
                    st.write("**Perbandingan Akurasi/F1 Semua Model:**")
                    results.plot(kind="leaderboard", ax=ax_roc)
                    st.pyplot(fig_roc)
                finally:
                    plt.close(fig_roc)
                st.info("💡 **Cara Membaca:** Garis kurva yang semakin mendekati sudut kiri-atas menunjukkan kemampuan model yang semakin sempurna dalam membedakan kelas.")

        elif results.task == "regression":
            st.markdown(f"#### 🎯 Evaluasi Model: `{selected_model.upper()}`")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.write("**Plot Residual (Sebaran Galat Error):**")
                fig_res, ax_res = plt.subplots(figsize=(5.5, 4.2))
                try:
                    results.plot(kind="residuals", ax=ax_res, model_name=selected_model)
                    st.pyplot(fig_res)
                except Exception as e:
                    st.warning(f"Gagal menggambar residual: {e}")
                finally:
                    plt.close(fig_res)
                st.info("💡 **Cara Membaca:** Titik-titik yang menyebar acak dan merata di sekitar garis horizontal 0 menunjukkan model yang sehat dan bebas bias (*homoskedastis*).")

            with col_r2:
                st.write("**Scatter Plot Nilai Aktual vs Prediksi:**")
                fig_avp, ax_avp = plt.subplots(figsize=(5.5, 4.2))
                try:
                    results.plot(kind="actual_vs_predicted", ax=ax_avp, model_name=selected_model)
                    st.pyplot(fig_avp)
                except Exception as e:
                    st.warning(f"Gagal menggambar scatter prediksi: {e}")
                finally:
                    plt.close(fig_avp)
                st.info("💡 **Cara Membaca:** Semakin dekat titik-titik data ke garis putus-putus merah ($y=x$), semakin presisi prediksi model.")

        elif results.task == "clustering":
            st.markdown("#### 🔍 Evaluasi Sebaran Klaster")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.write("**Proyeksi 2D Klaster (PCA):**")
                fig_cl, ax_cl = plt.subplots(figsize=(5.5, 4.2))
                try:
                    results.plot(kind="clusters", ax=ax_cl)
                    st.pyplot(fig_cl)
                except Exception as e:
                    st.warning(f"Gagal menggambar klaster: {e}")
                finally:
                    plt.close(fig_cl)
            with col_c2:
                st.write("**Perbandingan Silhouette Score:**")
                fig_lb, ax_lb = plt.subplots(figsize=(5.5, 4.2))
                results.plot(kind="leaderboard", ax=ax_lb)
                st.pyplot(fig_lb)
                plt.close(fig_lb)

        elif results.task == "association":
            st.markdown("#### 🛒 Evaluasi Aturan Asosiasi Transaksi")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.write("**Scatter Plot Aturan (Support vs Confidence vs Lift):**")
                fig_ar, ax_ar = plt.subplots(figsize=(5.5, 4.2))
                try:
                    results.plot(kind="rules", ax=ax_ar)
                    st.pyplot(fig_ar)
                except Exception as e:
                    st.warning(f"Gagal menggambar aturan: {e}")
                finally:
                    plt.close(fig_ar)
            with col_a2:
                st.write("**Jumlah Aturan yang Dihasilkan:**")
                fig_lb, ax_lb = plt.subplots(figsize=(5.5, 4.2))
                results.plot(kind="leaderboard", ax=ax_lb)
                st.pyplot(fig_lb)
                plt.close(fig_lb)

        elif results.task == "anomaly":
            st.markdown("#### 🚨 Evaluasi Deteksi Anomali")
            col_an1, col_an2 = st.columns(2)
            with col_an1:
                st.write("**Proyeksi Titik Anomali vs Normal (2D PCA):**")
                fig_an, ax_an = plt.subplots(figsize=(5.5, 4.2))
                try:
                    results.plot(kind="anomalies", ax=ax_an)
                    st.pyplot(fig_an)
                except Exception as e:
                    st.warning(f"Gagal menggambar anomali: {e}")
                finally:
                    plt.close(fig_an)
            with col_an2:
                st.write("**Persentase Anomali yang Terdeteksi Tiap Algoritma:**")
                fig_lb, ax_lb = plt.subplots(figsize=(5.5, 4.2))
                results.plot(kind="leaderboard", ax=ax_lb)
                st.pyplot(fig_lb)
                plt.close(fig_lb)

        st.markdown("---")
        
        # 3. Grafik Perbandingan Semua Algoritma (Leaderboard Bar Chart)
        st.markdown("#### 🏆 Perbandingan Performa Seluruh Algoritma (Leaderboard)")
        fig_board, ax_board = plt.subplots(figsize=(8.5, 3.8))
        try:
            results.plot(kind="leaderboard", ax=ax_board)
            st.pyplot(fig_board)
        except Exception as e:
            st.warning(f"Gagal menggambar leaderboard: {e}")
        finally:
            plt.close(fig_board)

        # 4. Diagram Alur Pipeline Data Mining
        st.markdown("#### 🏗️ Diagram Alur Pipeline Data Mining (Arsitektur)")
        fig_pipe, ax_pipe = plt.subplots(figsize=(9, 2.6))
        try:
            results.plot(kind="pipeline", ax=ax_pipe)
            st.pyplot(fig_pipe)
        except Exception as e:
            st.warning(f"Gagal membuat diagram pipeline: {e}")
        finally:
            plt.close(fig_pipe)
    else:
        st.info("Tekan tombol **'🚀 Jalankan Penambangan Data Sekarang'** pada bilah sisi kiri terlebih dahulu untuk melihat seluruh visualisasi.")

# ----------------- TAB 4: KESIMPULAN ILMIAH -----------------
with tab_summary:
    st.subheader("Kesimpulan Naratif & Interpretasi Ilmiah")
    results = st.session_state.results
    if results is not None:
        st.markdown(
            """
            > **Penjelasan Otomatis Berbahasa Indonesia:** Bagian ini otomatis menganalisis stabilitas model,
            mendeteksi ada/tidaknya indikasi *overfitting*, menginterpretasikan nilai metrik secara lugas,
            serta memberikan draf kalimat yang bisa langsung disalin ke Bab IV Skripsi atau naskah publikasi.
            """
        )
        narasi = results.summary()
        st.text_area("Draf Pembahasan Skripsi Otomatis:", narasi, height=360)
    else:
        st.info("Jalankan pemodelan terlebih dahulu untuk menghasilkan kesimpulan naratif otomatis.")
        
    st.markdown("---")
    st.subheader("📚 Kamus Konsep Data Mining Interaktif")
    selected_concept = st.selectbox("Pilih Konsep yang Ingin Dipelajari:", list_concepts())
    if selected_concept:
        st.text(explain_concept(selected_concept))

# ----------------- TAB 5: AUTO TUNING -----------------
with tab_tuning:
    st.subheader("⚡ Penyetelan Hyperparameter Otomatis")
    results = st.session_state.results
    if results is not None:
        if results.task not in {"classification", "regression"}:
            st.info("Penyetelan hyperparameter otomatis saat ini khusus untuk tugas Klasifikasi dan Regresi.")
        else:
            st.write(
                f"Optimalkan algoritma pemenang **{str(results.model_name).upper()}** dengan mencari kombinasi parameter "
                "terbaik secara otomatis tanpa kebocoran data (*zero data leakage*)."
            )
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                tune_method = st.selectbox("Metode Pencarian:", ["random", "grid"], help="Random Search lebih cepat; Grid Search menguji seluruh kombinasi.")
            with col_t2:
                n_iter_tune = st.slider("Jumlah Iterasi Pengujian Parameter:", 4, 16, 6)
                
            if st.button("🚀 Jalankan Auto-Tuning Sekarang", type="primary"):
                with st.spinner("⏳ Menjalankan penyetelan hyperparameter..."):
                    try:
                        tuned = results.tune(method=tune_method, n_iter=n_iter_tune)
                        st.session_state.tuned_results = tuned
                        st.success("Penyetelan parameter selesai!")
                    except Exception as e:
                        st.error(f"Gagal melakukan tuning: {e}")
                        
            if st.session_state.tuned_results is not None:
                tuned = st.session_state.tuned_results
                st.markdown("##### Ringkasan Hasil Penyetelan:")
                st.text(tuned.summary())
                st.write("**Tabel Perbandingan Sebelum vs Sesudah:**")
                st.dataframe(tuned.comparison, use_container_width=True)
                st.write("**Parameter Terbaik yang Ditemukan:**")
                st.json(tuned.best_params)
    else:
        st.info("Jalankan pemodelan terlebih dahulu pada bilah sisi.")

# ----------------- TAB 6: EKSPOR SKRIPSI / LATEX -----------------
with tab_export:
    st.subheader("🎓 Ekspor Hasil untuk Skripsi & Publikasi Jurnal")
    results = st.session_state.results
    if results is not None:
        st.write("Salin tabel terformat di bawah ini langsung ke naskah skripsi Microsoft Word, Google Docs, atau Overleaf LaTeX.")
        
        st.markdown("##### 1. Format Markdown Baku (Bab IV Skripsi / Dokumen):")
        academic_table = results.to_academic_table()
        st.code(academic_table, language="markdown")
        
        st.markdown("##### 2. Format Tabel LaTeX Baku (IEEE / Springer / SINTA / Overleaf):")
        latex_code = results.to_latex()
        st.code(latex_code, language="latex")
        
        st.markdown("---")
        st.subheader("💾 Unduh Artefak Model & Laporan")
        c_down1, c_down2 = st.columns(2)
        with c_down1:
            try:
                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
                    results.save(tmp_file.name)
                    with open(tmp_file.name, "rb") as f_in:
                        joblib_bytes = f_in.read()
                st.download_button(
                    "📦 Unduh Model Pipeline (.joblib)",
                    joblib_bytes,
                    f"model_{results.task}.joblib",
                    "application/octet-stream",
                    help="Model pipeline siap dideploy untuk memprediksi data baru secara langsung.",
                )
            except Exception as e:
                st.warning(f"Unduhan model joblib: {e}")
        with c_down2:
            try:
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_html:
                    results.report(tmp_html.name)
                    with open(tmp_html.name, "rb") as f_html:
                        html_bytes = f_html.read()
                st.download_button(
                    "🌐 Unduh Laporan Web Lengkap (.html)",
                    html_bytes,
                    f"laporan_{results.task}.html",
                    "text/html",
                    help="Laporan HTML mandiri yang berisi seluruh leaderboard, konfigurasi, dan kesimpulan naratif.",
                )
            except Exception as e:
                st.warning(f"Unduhan laporan html: {e}")
    else:
        st.info("Jalankan pemodelan terlebih dahulu untuk mengekspor hasil ke skripsi atau LaTeX.")

# ----------------- TAB 7: UJI PREDIKSI LANGSUNG -----------------
with tab_predict:
    st.subheader("🔮 Uji Prediksi Data Baru Secara Langsung")
    results = st.session_state.results
    if results is not None and results.task in {"classification", "regression"}:
        st.write("Masukkan nilai fitur di bawah untuk melihat hasil prediksi model secara instan:")
        
        active_pipeline = results.best_model
        if st.session_state.tuned_results is not None:
            active_pipeline = st.session_state.tuned_results.best_estimator
            st.info("Menggunakan model hasil penyetelan optimal (Tuned Pipeline).")
            
        # Dapatkan fitur yang digunakan (selain target)
        df_fitur = st.session_state.df.drop(columns=[target_col])
        input_data = {}
        
        cols_input = st.columns(min(3, df_fitur.shape[1]))
        for idx, col in enumerate(df_fitur.columns):
            with cols_input[idx % len(cols_input)]:
                if pd.api.types.is_numeric_dtype(df_fitur[col]):
                    val_mean = float(df_fitur[col].mean())
                    val_min = float(df_fitur[col].min())
                    val_max = float(df_fitur[col].max())
                    input_data[col] = st.number_input(f"{col}:", value=val_mean, min_value=val_min, max_value=val_max)
                else:
                    unique_vals = list(df_fitur[col].dropna().unique())
                    input_data[col] = st.selectbox(f"{col}:", unique_vals)
                    
        if st.button("🔮 Hitung Prediksi Sekarang", type="primary"):
            try:
                single_row_df = pd.DataFrame([input_data])
                y_pred = active_pipeline.predict(single_row_df)
                st.success(f"### 🎉 Hasil Prediksi: `{y_pred[0]}`")
            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")
    else:
        st.info("Uji prediksi interaktif tersedia untuk tugas Klasifikasi dan Regresi setelah model dilatih.")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Dibuat untuk memudahkan mahasiswa & peneliti Indonesia dalam belajar dan mempublikasikan karya ilmiah Data Mining. "
    "Didukung oleh pustaka <b>AutoDataMining</b> & <b>scikit-learn</b>.</small></center>",
    unsafe_allow_html=True,
)
