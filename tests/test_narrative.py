"""Tests for automated narrative conclusions (results.summary and results.kesimpulan)."""

import pandas as pd
from sklearn.datasets import make_blobs, make_classification, make_regression

from autodatamining import AutoDataMining


def test_narrative_classification():
    X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y

    adm = AutoDataMining(task="classification", target="target", models=["logistic_regression"], random_state=42)
    results = adm.fit(df)

    summary = results.summary()
    assert "KESIMPULAN & RINGKASAN EVALUASI PENELITIAN" in summary
    assert "Klasifikasi (Classification)" in summary
    assert "PERFORMA MODEL TERBAIK" in summary
    assert "ANALISIS GENERALISASI" in summary
    assert "DRAFT KALIMAT PEMBAHASAN UNTUK SKRIPSI / PAPER" in summary

    # Verify alias kesimpulan()
    assert results.kesimpulan() == summary


def test_narrative_regression():
    X, y = make_regression(n_samples=100, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y

    adm = AutoDataMining(task="regression", target="target", models=["linear_regression"], random_state=42)
    results = adm.fit(df)

    summary = results.summary()
    assert "Regresi (Regression)" in summary
    assert "Koefisien Determinasi (R²" in summary
    assert "MAE" in summary
    assert "RMSE" in summary


def test_narrative_clustering():
    X, _ = make_blobs(n_samples=90, centers=3, random_state=42)
    df = pd.DataFrame(X, columns=["x", "y"])

    adm = AutoDataMining(task="clustering", models=["kmeans"], random_state=42)
    results = adm.fit(df)

    summary = results.summary()
    assert "Pengelompokan (Clustering)" in summary
    assert "Jumlah Klaster Terbentuk" in summary
    assert "Skor Silhouette" in summary


def test_narrative_anomaly():
    X, _ = make_blobs(n_samples=90, centers=1, random_state=42)
    df = pd.DataFrame(X, columns=["x", "y"])

    adm = AutoDataMining(task="anomaly", models=["isolation_forest"], random_state=42)
    results = adm.fit(df)

    summary = results.summary()
    assert "Deteksi Anomali (Anomaly Detection)" in summary
    assert "TEMUAN DATA ANOMALI" in summary


def test_narrative_association():
    transaksi = [
        ["roti", "susu", "mentega"],
        ["roti", "mentega"],
        ["susu", "roti", "keju"],
        ["biskuit", "kopi"],
        ["roti", "mentega", "keju"],
    ]

    adm = AutoDataMining(task="association", min_support=0.3, min_confidence=0.5)
    results = adm.fit(transaksi)

    summary = results.summary()
    assert "Aturan Asosiasi (Association Rule Mining)" in summary
    assert "Frequent Itemsets Ditemukan" in summary
    assert "ATURAN PALING BERPENGARUH" in summary
