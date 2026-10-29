"""Tests for association rule mining (TransactionEncoder, apriori, fpgrowth, association_rules)."""

import pandas as pd
import pytest

from autodatamining import (
    AutoDataMining,
    TransactionEncoder,
    apriori,
    association_rules,
    fpgrowth,
)
from autodatamining.exceptions import DataValidationError


@pytest.fixture
def transactions():
    return [
        ["milk", "bread", "butter"],
        ["bread", "butter"],
        ["milk", "bread"],
        ["milk", "diaper", "beer"],
        ["bread", "butter", "cookie"],
    ]


@pytest.fixture
def transaction_df(transactions):
    te = TransactionEncoder()
    return te.fit_transform(transactions)


def test_transaction_encoder(transactions):
    te = TransactionEncoder()
    df = te.fit_transform(transactions)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert set(df.columns) == {"beer", "bread", "butter", "cookie", "diaper", "milk"}
    assert df.loc[0, "milk"] is True or df.loc[0, "milk"] == 1
    assert df.loc[0, "beer"] is False or df.loc[0, "beer"] == 0


def test_apriori_and_fpgrowth_match(transaction_df):
    # Both algorithms must discover the exact same frequent itemsets with the same supports
    res_apriori = apriori(transaction_df, min_support=0.4)
    res_fpgrowth = fpgrowth(transaction_df, min_support=0.4)

    assert not res_apriori.empty
    assert not res_fpgrowth.empty
    assert len(res_apriori) == len(res_fpgrowth)

    # Convert to dictionary for easy comparison
    dict_apriori = {row["itemsets"]: round(row["support"], 4) for _, row in res_apriori.iterrows()}
    dict_fpgrowth = {row["itemsets"]: round(row["support"], 4) for _, row in res_fpgrowth.iterrows()}

    assert dict_apriori == dict_fpgrowth
    # Check known frequent itemset: {bread, butter} appears in 3 out of 5 transactions (support = 0.6)
    assert dict_apriori[frozenset(["bread", "butter"])] == 0.6
    # {bread} appears in 4 of 5 (support = 0.8)
    assert dict_apriori[frozenset(["bread"])] == 0.8


def test_max_len_constraint(transaction_df):
    itemsets_all = fpgrowth(transaction_df, min_support=0.2)
    itemsets_max1 = fpgrowth(transaction_df, min_support=0.2, max_len=1)
    assert all(len(itemset) == 1 for itemset in itemsets_max1["itemsets"])
    assert len(itemsets_all) >= len(itemsets_max1)


def test_association_rules_metrics(transaction_df):
    frequent = fpgrowth(transaction_df, min_support=0.4)
    rules = association_rules(frequent, metric="confidence", min_threshold=0.5)

    assert not rules.empty
    expected_cols = {
        "antecedents",
        "consequents",
        "antecedent support",
        "consequent support",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
    }
    assert expected_cols.issubset(rules.columns)

    for _, rule in rules.iterrows():
        ant_supp = rule["antecedent support"]
        con_supp = rule["consequent support"]
        supp = rule["support"]
        conf = rule["confidence"]
        lift = rule["lift"]
        assert pytest.approx(supp / ant_supp, rel=1e-3) == conf
        assert pytest.approx(conf / con_supp, rel=1e-3) == lift
        assert rule["confidence"] >= 0.5


def test_autodatamining_association_task(transactions):
    # End-to-end integration with AutoDataMining on raw transactions
    adm = AutoDataMining(
        task="association",
        min_support=0.4,
        min_confidence=0.5,
    )
    results = adm.fit(transactions)

    assert results.task == "association"
    assert results.best_model is None
    assert results.model_name in {"fpgrowth", "apriori"}
    assert results.rules is not None
    assert results.frequent_itemsets is not None
    assert len(results.leaderboard) == 2
    assert "execution_seconds" in results.leaderboard.columns
    assert "rules_count" in results.leaderboard.columns
    assert (results.leaderboard["status"] == "ok").all()


def test_autodatamining_association_on_dataframe(transaction_df):
    adm = AutoDataMining(
        task="association_rules",  # test alias
        min_support=0.3,
        min_confidence=0.4,
        models=["fpgrowth"],
    )
    results = adm.fit(transaction_df)
    assert len(results.leaderboard) == 1
    assert results.model_name == "fpgrowth"
    assert not results.frequent_itemsets.empty


def test_association_invalid_inputs():
    with pytest.raises(DataValidationError):
        apriori(pd.DataFrame(), min_support=0.5)

    with pytest.raises(DataValidationError):
        fpgrowth(pd.DataFrame([[]]), min_support=1.5)

    with pytest.raises(DataValidationError):
        association_rules(pd.DataFrame(), metric="invalid_metric")
