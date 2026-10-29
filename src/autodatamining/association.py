"""Association rule mining using Apriori and FP-Growth algorithms."""

from collections import defaultdict
from itertools import combinations
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .exceptions import DataValidationError


class TransactionEncoder:
    """Transform an iterable of transaction sequences into a boolean DataFrame."""

    def __init__(self):
        self.columns_ = []

    def fit(self, transactions: Iterable[Iterable[Any]]) -> "TransactionEncoder":
        unique_items = set()
        for transaction in transactions:
            unique_items.update(transaction)
        self.columns_ = sorted(unique_items, key=lambda x: str(x))
        return self

    def transform(self, transactions: Iterable[Iterable[Any]]) -> pd.DataFrame:
        if not self.columns_:
            raise DataValidationError("TransactionEncoder has not been fitted yet.")
        item_to_idx = {item: idx for idx, item in enumerate(self.columns_)}
        rows = []
        for transaction in transactions:
            row = np.zeros(len(self.columns_), dtype=bool)
            for item in transaction:
                if item in item_to_idx:
                    row[item_to_idx[item]] = True
            rows.append(row)
        return pd.DataFrame(rows, columns=self.columns_)

    def fit_transform(self, transactions: Iterable[Iterable[Any]]) -> pd.DataFrame:
        return self.fit(transactions).transform(transactions)

    def to_dataframe(self, array_or_df: Any) -> pd.DataFrame:
        """Convert a boolean array or return DataFrame with encoder columns."""
        if isinstance(array_or_df, pd.DataFrame):
            return array_or_df
        return pd.DataFrame(array_or_df, columns=self.columns_)


def _prepare_transaction_matrix(df_or_transactions):
    """Normalize input into a boolean DataFrame and validate shape."""
    if isinstance(df_or_transactions, pd.DataFrame):
        df = df_or_transactions.copy()
        # Verify if values are boolean-like or 0/1
        if df.empty:
            raise DataValidationError("Input DataFrame cannot be empty.")
        # Convert non-boolean columns if they are binary numeric
        is_bool = df.dtypes == bool
        if not is_bool.all():
            try:
                df = df.astype(bool)
            except Exception as exc:
                raise DataValidationError(f"Could not convert input data to boolean matrix: {exc}") from exc
        return df
    if isinstance(df_or_transactions, (list, tuple)):
        encoder = TransactionEncoder()
        return encoder.fit_transform(df_or_transactions)
    raise DataValidationError("Input must be a boolean DataFrame or an iterable of transaction item lists.")


def apriori(
    df_or_transactions,
    min_support: float = 0.5,
    use_colnames: bool = True,
    max_len: int = None,
) -> pd.DataFrame:
    """Find frequent itemsets using the classic Apriori algorithm.

    Parameters
    ----------
    df_or_transactions : DataFrame or list of lists
        One-hot boolean DataFrame or list of transaction lists.
    min_support : float, default=0.5
        Minimum fraction of transactions an itemset must appear in (between 0 and 1).
    use_colnames : bool, default=True
        Whether to use column names as items in itemsets instead of column indices.
    max_len : int, optional
        Maximum length of itemsets to discover.
    """
    if not (0 < min_support <= 1.0):
        raise DataValidationError("min_support must be between 0 (exclusive) and 1 (inclusive).")

    df = _prepare_transaction_matrix(df_or_transactions)
    n_transactions = len(df)
    columns = list(df.columns)
    matrix = df.values.astype(bool)

    # 1-itemsets
    col_supports = matrix.mean(axis=0)
    frequent_1 = {
        (i,): col_supports[i]
        for i in range(len(columns))
        if col_supports[i] >= min_support
    }

    all_frequent = dict(frequent_1)
    current_frequent = set(frequent_1.keys())
    k = 2

    while current_frequent and (max_len is None or k <= max_len):
        # Candidate generation: join step
        candidates = set()
        current_list = sorted(current_frequent)
        for i in range(len(current_list)):
            for j in range(i + 1, len(current_list)):
                itemset1, itemset2 = current_list[i], current_list[j]
                if itemset1[:-1] == itemset2[:-1]:
                    candidate = tuple(sorted(set(itemset1) | set(itemset2)))
                    # Prune step: check all subsets of length k-1
                    subsets_frequent = True
                    for sub in combinations(candidate, k - 1):
                        if sub not in current_frequent:
                            subsets_frequent = False
                            break
                    if subsets_frequent:
                        candidates.add(candidate)

        if not candidates:
            break

        # Support counting
        next_frequent = {}
        for candidate in candidates:
            # Vectorized bitwise AND across candidate item columns
            sub_matrix = matrix[:, list(candidate)]
            supp = np.all(sub_matrix, axis=1).sum() / n_transactions
            if supp >= min_support:
                next_frequent[candidate] = supp

        all_frequent.update(next_frequent)
        current_frequent = set(next_frequent.keys())
        k += 1

    records = []
    for itemset_indices, supp in all_frequent.items():
        if use_colnames:
            itemset = frozenset(columns[idx] for idx in itemset_indices)
        else:
            itemset = frozenset(itemset_indices)
        records.append({"support": supp, "itemsets": itemset})

    if not records:
        return pd.DataFrame(columns=["support", "itemsets"])

    res_df = pd.DataFrame(records).sort_values(by="support", ascending=False).reset_index(drop=True)
    return res_df


class _FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None


def fpgrowth(
    df_or_transactions,
    min_support: float = 0.5,
    use_colnames: bool = True,
    max_len: int = None,
) -> pd.DataFrame:
    """Find frequent itemsets using the FP-Growth (Frequent Pattern Tree) algorithm.

    Parameters
    ----------
    df_or_transactions : DataFrame or list of lists
        One-hot boolean DataFrame or list of transaction lists.
    min_support : float, default=0.5
        Minimum fraction of transactions an itemset must appear in (between 0 and 1).
    use_colnames : bool, default=True
        Whether to use column names as items in itemsets instead of column indices.
    max_len : int, optional
        Maximum length of itemsets to discover.
    """
    if not (0 < min_support <= 1.0):
        raise DataValidationError("min_support must be between 0 (exclusive) and 1 (inclusive).")

    df = _prepare_transaction_matrix(df_or_transactions)
    n_transactions = len(df)
    min_count = min_support * n_transactions
    columns = list(df.columns)
    matrix = df.values.astype(bool)

    # 1. Calculate item counts
    counts = matrix.sum(axis=0)
    frequent_items = [
        i for i in range(len(columns)) if counts[i] >= min_count
    ]
    # Sort frequent items descending by count, then ascending by index for deterministic order
    frequent_items.sort(key=lambda x: (-counts[x], x))
    order_map = {item: rank for rank, item in enumerate(frequent_items)}

    # 2. Build transactions list with only frequent items sorted
    transactions = []
    for row in matrix:
        items_present = [idx for idx in np.where(row)[0] if idx in order_map]
        if items_present:
            items_present.sort(key=lambda idx: order_map[idx])
            transactions.append(items_present)

    # 3. Build FP-Tree
    root = _FPNode(item=None, count=0, parent=None)
    header_table = {item: None for item in frequent_items}

    for trans in transactions:
        curr_node = root
        for item in trans:
            if item in curr_node.children:
                curr_node.children[item].count += 1
            else:
                new_node = _FPNode(item=item, count=1, parent=curr_node)
                curr_node.children[item] = new_node
                # Link in header table
                if header_table[item] is None:
                    header_table[item] = new_node
                else:
                    head = header_table[item]
                    while head.node_link is not None:
                        head = head.node_link
                    head.node_link = new_node
            curr_node = curr_node.children[item]

    # 4. Mine FP-Tree recursively
    results = []

    def _mine_tree(header, prefix, current_len):
        # Process items in reverse order of frequency (bottom-up)
        sorted_header_items = sorted(header.keys(), key=lambda x: order_map[x], reverse=True)
        for item in sorted_header_items:
            head_node = header[item]
            item_count = 0
            node = head_node
            while node is not None:
                item_count += node.count
                node = node.node_link

            new_prefix = prefix | {item}
            results.append((new_prefix, item_count / n_transactions))

            if max_len is not None and current_len >= max_len:
                continue

            # Find conditional pattern base
            conditional_patterns = []
            node = head_node
            while node is not None:
                path = []
                p = node.parent
                while p is not None and p.item is not None:
                    path.append(p.item)
                    p = p.parent
                if path:
                    conditional_patterns.append((path[::-1], node.count))
                node = node.node_link

            # Build conditional tree
            cond_counts = defaultdict(int)
            for path, cnt in conditional_patterns:
                for p_item in path:
                    cond_counts[p_item] += cnt

            cond_frequent = {it: cnt for it, cnt in cond_counts.items() if cnt >= min_count}
            if not cond_frequent:
                continue

            cond_root = _FPNode(item=None, count=0, parent=None)
            cond_header = {it: None for it in cond_frequent}
            for path, cnt in conditional_patterns:
                filtered_path = [it for it in path if it in cond_frequent]
                filtered_path.sort(key=lambda idx: order_map[idx])
                curr_cond = cond_root
                for p_item in filtered_path:
                    if p_item in curr_cond.children:
                        curr_cond.children[p_item].count += cnt
                    else:
                        new_cond_node = _FPNode(item=p_item, count=cnt, parent=curr_cond)
                        curr_cond.children[p_item] = new_cond_node
                        if cond_header[p_item] is None:
                            cond_header[p_item] = new_cond_node
                        else:
                            head = cond_header[p_item]
                            while head.node_link is not None:
                                head = head.node_link
                            head.node_link = new_cond_node
                    curr_cond = curr_cond.children[p_item]

            _mine_tree(cond_header, new_prefix, current_len + 1)

    _mine_tree(header_table, set(), 1)

    records = []
    for itemset_indices, supp in results:
        if use_colnames:
            itemset = frozenset(columns[idx] for idx in itemset_indices)
        else:
            itemset = frozenset(itemset_indices)
        records.append({"support": supp, "itemsets": itemset})

    if not records:
        return pd.DataFrame(columns=["support", "itemsets"])

    res_df = pd.DataFrame(records).sort_values(by="support", ascending=False).reset_index(drop=True)
    return res_df


def association_rules(
    df_frequent_itemsets: pd.DataFrame,
    metric: str = "confidence",
    min_threshold: float = 0.5,
) -> pd.DataFrame:
    """Generate association rules from frequent itemsets with key data mining metrics.

    Metrics calculated:
    - antecedent support: supp(A)
    - consequent support: supp(C)
    - support: supp(A U C)
    - confidence: supp(A U C) / supp(A)
    - lift: conf(A -> C) / supp(C)
    - leverage: supp(A U C) - (supp(A) * supp(C))
    - conviction: (1 - supp(C)) / (1 - conf(A -> C))

    Parameters
    ----------
    df_frequent_itemsets : pd.DataFrame
        DataFrame containing 'support' and 'itemsets' columns.
    metric : str, default='confidence'
        Metric to filter rules ('confidence', 'lift', 'support', 'leverage', 'conviction').
    min_threshold : float, default=0.5
        Minimum threshold value for the specified metric.
    """
    valid_metrics = {"confidence", "lift", "support", "leverage", "conviction"}
    if metric not in valid_metrics:
        raise DataValidationError(f"Invalid metric {metric!r}; choose from {sorted(valid_metrics)}")

    if df_frequent_itemsets.empty:
        return pd.DataFrame(
            columns=[
                "antecedents",
                "consequents",
                "antecedent support",
                "consequent support",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
            ]
        )

    # Build itemset -> support lookup
    support_dict = {
        frozenset(row["itemsets"]): row["support"]
        for _, row in df_frequent_itemsets.iterrows()
    }

    rules = []
    for itemset, itemset_support in support_dict.items():
        if len(itemset) < 2:
            continue
        # Generate non-empty proper subsets as antecedents
        for r in range(1, len(itemset)):
            for antecedent_tuple in combinations(itemset, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                if antecedent not in support_dict or consequent not in support_dict:
                    continue

                ant_supp = support_dict[antecedent]
                con_supp = support_dict[consequent]
                supp = itemset_support
                confidence = supp / ant_supp if ant_supp > 0 else 0.0
                lift = confidence / con_supp if con_supp > 0 else 0.0
                leverage = supp - (ant_supp * con_supp)
                if confidence >= 1.0:
                    conviction = np.inf
                else:
                    conviction = (1.0 - con_supp) / (1.0 - confidence)

                rule_metrics = {
                    "antecedents": antecedent,
                    "consequents": consequent,
                    "antecedent support": ant_supp,
                    "consequent support": con_supp,
                    "support": supp,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": leverage,
                    "conviction": conviction,
                }

                if rule_metrics[metric] >= min_threshold:
                    rules.append(rule_metrics)

    if not rules:
        return pd.DataFrame(
            columns=[
                "antecedents",
                "consequents",
                "antecedent support",
                "consequent support",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
            ]
        )

    res_df = pd.DataFrame(rules)
    # Sort by lift descending, then confidence descending
    res_df = res_df.sort_values(by=["lift", "confidence"], ascending=[False, False]).reset_index(drop=True)
    return res_df
