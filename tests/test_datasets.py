"""Tests for built-in educational datasets (load_dataset and list_datasets)."""

import pandas as pd
import pytest

from autodatamining import AutoDataMining, list_datasets, load_dataset
from autodatamining.exceptions import DataValidationError


def test_list_datasets():
    datasets = list_datasets()
    assert isinstance(datasets, list)
    assert "kelulusan_mahasiswa" in datasets
    assert "harga_rumah" in datasets
    assert "segmentasi_nasabah" in datasets
    assert "keranjang_belanja" in datasets
    assert "deteksi_penipuan" in datasets


def test_load_graduation_dataset():
    df = load_dataset("kelulusan_mahasiswa")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 120
    assert "ipk" in df.columns
    assert "lulus_tepat_waktu" in df.columns
    assert set(df["lulus_tepat_waktu"].unique()) == {"Ya", "Tidak"}


def test_load_housing_dataset():
    df = load_dataset("harga_rumah")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "harga_juta" in df.columns
    assert df["harga_juta"].dtype in [float, int]


def test_load_customers_dataset():
    df = load_dataset("segmentasi_nasabah")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "pendapatan_tahunan_juta" in df.columns


def test_load_basket_dataset():
    basket = load_dataset("keranjang_belanja")
    assert isinstance(basket, list)
    assert len(basket) == 20
    assert isinstance(basket[0], list)


def test_load_fraud_dataset():
    df = load_dataset("deteksi_penipuan")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "nominal_transaksi" in df.columns


def test_dataset_aliases():
    df1 = load_dataset("student")
    assert "lulus_tepat_waktu" in df1.columns
    b1 = load_dataset("supermarket")
    assert isinstance(b1, list)


def test_invalid_dataset():
    with pytest.raises(DataValidationError):
        load_dataset("unknown_dataset_xyz")


def test_end_to_end_with_builtin_dataset():
    df = load_dataset("kelulusan_mahasiswa")
    adm = AutoDataMining(
        task="classification",
        target="lulus_tepat_waktu",
        models=["logistic_regression"],
        random_state=42,
    )
    results = adm.fit(df)
    assert results.model_name == "logistic_regression"
    assert "accuracy" in results.metrics
