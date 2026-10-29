"""Tests for academic table and LaTeX export (to_academic_table and to_latex)."""

import pandas as pd
from sklearn.datasets import make_classification

from autodatamining import AutoDataMining


def test_to_academic_table_and_latex():
    X, y = make_classification(n_samples=80, n_features=4, n_informative=2, n_redundant=0, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y

    adm = AutoDataMining(
        task="classification",
        target="target",
        models=["logistic_regression", "random_forest"],
        random_state=42,
    )
    results = adm.fit(df)

    # 1. Test DataFrame format
    df_table = results.to_academic_table(format="dataframe")
    assert isinstance(df_table, pd.DataFrame)
    assert "Peringkat" in df_table.columns
    assert "Algoritma" in df_table.columns
    assert "CV Data Latih" in df_table.columns

    # 2. Test Markdown format
    md_table = results.to_academic_table(format="markdown")
    assert isinstance(md_table, str)
    assert "Peringkat" in md_table
    assert "Algoritma" in md_table

    # 3. Test LaTeX export
    latex = results.to_latex(caption="Tabel Hasil Pengujian", label="tab:hasil_klasifikasi")
    assert isinstance(latex, str)
    assert "\\begin{table}" in latex
    assert "\\caption{Tabel Hasil Pengujian}" in latex
    assert "\\label{tab:hasil_klasifikasi}" in latex
    assert "\\toprule" in latex
    assert "\\bottomrule" in latex
    assert "\\end{table}" in latex
