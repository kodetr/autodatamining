"""Built-in offline educational datasets for hands-on learning and research baselines."""

import numpy as np
import pandas as pd

from .exceptions import DataValidationError


def list_datasets() -> list[str]:
    """Return a list of available built-in dataset names."""
    return [
        "kelulusan_mahasiswa",
        "harga_rumah",
        "segmentasi_nasabah",
        "keranjang_belanja",
        "deteksi_penipuan",
    ]


def load_dataset(name: str):
    """Load a built-in offline educational dataset.

    Available datasets:
    - 'kelulusan_mahasiswa': Tabular classification (target: 'lulus_tepat_waktu')
    - 'harga_rumah': Tabular regression (target: 'harga_juta')
    - 'segmentasi_nasabah': Tabular clustering (4 numerical features)
    - 'keranjang_belanja': List of transactions for association rule mining
    - 'deteksi_penipuan': Tabular anomaly detection (4 transaction behavioral features)

    Parameters
    ----------
    name : str
        The dataset identifier.
    """
    key = str(name).strip().lower()
    aliases = {
        "graduation": "kelulusan_mahasiswa",
        "kelulusan": "kelulusan_mahasiswa",
        "student": "kelulusan_mahasiswa",
        "housing": "harga_rumah",
        "house": "harga_rumah",
        "rumah": "harga_rumah",
        "customer": "segmentasi_nasabah",
        "nasabah": "segmentasi_nasabah",
        "clustering": "segmentasi_nasabah",
        "supermarket": "keranjang_belanja",
        "market": "keranjang_belanja",
        "transaksi": "keranjang_belanja",
        "fraud": "deteksi_penipuan",
        "anomali": "deteksi_penipuan",
        "penipuan": "deteksi_penipuan",
    }
    canonical = aliases.get(key, key)

    if canonical == "kelulusan_mahasiswa":
        return _make_graduation_dataset()
    if canonical == "harga_rumah":
        return _make_housing_dataset()
    if canonical == "segmentasi_nasabah":
        return _make_customers_dataset()
    if canonical == "keranjang_belanja":
        return _make_basket_dataset()
    if canonical == "deteksi_penipuan":
        return _make_fraud_dataset()

    available = ", ".join(list_datasets())
    raise DataValidationError(f"Dataset {name!r} not found. Available datasets: {available}")


def _make_graduation_dataset() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 120
    ipk = np.round(rng.uniform(2.4, 3.95, n), 2)
    sks = rng.randint(70, 94, n)
    kehadiran = rng.randint(60, 100, n)
    organisasi = rng.choice(["Ya", "Tidak"], n, p=[0.45, 0.55])

    # Realistic graduation probability formula
    score = (ipk - 2.0) * 1.8 + (sks - 70) * 0.05 + (kehadiran - 60) * 0.04
    score += (organisasi == "Ya") * 0.2 + rng.normal(0, 0.4, n)
    target = np.where(score > 2.2, "Ya", "Tidak")

    return pd.DataFrame({
        "ipk": ipk,
        "sks_semester_4": sks,
        "kehadiran_persen": kehadiran,
        "ikut_organisasi": organisasi,
        "lulus_tepat_waktu": target,
    })


def _make_housing_dataset() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 100
    luas_tanah = rng.randint(60, 300, n)
    luas_bangunan = np.clip(luas_tanah - rng.randint(10, 50, n), 36, 250)
    kamar_tidur = rng.randint(2, 6, n)
    jarak_kota = np.round(rng.uniform(2.0, 25.0, n), 1)

    # Realistic house price in millions IDR
    base_price = (luas_tanah * 3.5) + (luas_bangunan * 5.0) + (kamar_tidur * 40.0) - (jarak_kota * 12.0)
    harga = np.round(base_price + rng.normal(0, 45, n), 1)
    harga = np.maximum(harga, 250.0)

    return pd.DataFrame({
        "luas_tanah": luas_tanah,
        "luas_bangunan": luas_bangunan,
        "kamar_tidur": kamar_tidur,
        "jarak_pusat_kota_km": jarak_kota,
        "harga_juta": harga,
    })


def _make_customers_dataset() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    # 3 distinct customer clusters
    c1 = rng.multivariate_normal([35, 40, 25], [[15, 2, 2], [2, 10, 1], [2, 1, 8]], 35)
    c2 = rng.multivariate_normal([75, 85, 60], [[20, 5, 3], [5, 15, 2], [3, 2, 12]], 35)
    c3 = rng.multivariate_normal([120, 30, 95], [[25, 3, 4], [3, 12, 2], [4, 2, 15]], 30)

    data = np.vstack([c1, c2, c3])
    df = pd.DataFrame(data, columns=["pendapatan_tahunan_juta", "skor_pengeluaran_1_100", "saldo_rekening_juta"])
    df = df.round(1)
    df["usia"] = rng.randint(21, 62, len(df))
    return df[["usia", "pendapatan_tahunan_juta", "skor_pengeluaran_1_100", "saldo_rekening_juta"]]


def _make_basket_dataset() -> list[list[str]]:
    return [
        ["roti", "susu", "mentega"],
        ["roti", "mentega"],
        ["susu", "roti", "keju"],
        ["kopi", "gula"],
        ["kopi", "gula", "biskuit"],
        ["roti", "mentega", "keju"],
        ["susu", "kopi", "gula"],
        ["teh", "gula"],
        ["teh", "gula", "biskuit"],
        ["roti", "selai"],
        ["roti", "susu", "selai"],
        ["kopi", "roti", "mentega"],
        ["sabun", "sampo"],
        ["sabun", "sampo", "pasta_gigi"],
        ["susu", "biskuit"],
        ["roti", "mentega", "susu", "keju"],
        ["teh", "kopi", "gula"],
        ["pasta_gigi", "sabun"],
        ["roti", "keju"],
        ["susu", "keju"],
    ]


def _make_fraud_dataset() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n_normal = 95
    n_outliers = 5

    # Normal transactions
    nominal_norm = rng.uniform(25000, 1500000, n_normal)
    jarak_norm = rng.uniform(0.5, 30.0, n_normal)
    jam_norm = rng.randint(7, 22, n_normal)
    frek_norm = rng.randint(1, 6, n_normal)

    # Anomaly / Outlier transactions (huge amount, strange hour, far distance)
    nominal_out = rng.uniform(15000000, 35000000, n_outliers)
    jarak_out = rng.uniform(300.0, 1200.0, n_outliers)
    jam_out = rng.choice([2, 3, 4], n_outliers)
    frek_out = rng.randint(15, 30, n_outliers)

    nominal = np.concatenate([nominal_norm, nominal_out])
    jarak = np.concatenate([jarak_norm, jarak_out])
    jam = np.concatenate([jam_norm, jam_out])
    frek = np.concatenate([frek_norm, frek_out])

    df = pd.DataFrame({
        "nominal_transaksi": np.round(nominal, -3),
        "jarak_km": np.round(jarak, 1),
        "jam_transaksi": jam,
        "frekuensi_24jam": frek,
    })
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)
