"""Input and output validation for the TA class trading system.

Implements pre-trade and post-trade sanity checks based on the
Jaramillo & Manjunath (2012) Top Trading Cycles mechanism.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def compute_priority_order(individuals: List[str]) -> List[str]:
    """Derive tie-breaking priority order deterministically by sorting SHA-256 hex digests.

    Lower index in the returned list indicates higher priority.
    """
    return sorted(
        individuals,
        key=lambda s: hashlib.sha256(str(s).encode("utf-8")).hexdigest(),
    )


def validate_pre_trade(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Validate input DataFrame according to Pre-Trade sanity checks.

    Checks:
    1. Required columns 'individuals' and 'baseline' must exist.
    2. Exact set equality between class preference columns and baseline classes:
       set(class_columns) == set(df['baseline']).
    3. Unique individuals: No duplicate names/IDs in 'individuals'.
    4. No null, empty, or whitespace values in 'individuals' or 'baseline'.
    5. For each individual, numeric ranks must be positive integers forming a
       contiguous set {1, 2, ..., n} with no gaps. If gaps exist, raise ValueError.
    6. Missing or blank class cells for an individual are filled with n+1
       (including baseline if left blank).
    7. Tie-breaking priority order is derived deterministically via SHA-256.

    Returns:
        clean_df: Cleaned DataFrame with integer ranks and filled blanks.
        class_columns: List of validated class preference columns.
        priority_order: Deterministic priority ordering of individuals.
    """
    clean_df = df.copy()
    # Strip leading/trailing whitespace from column names
    clean_df.columns = [c.strip() if isinstance(c, str) else c for c in clean_df.columns]

    # Check 1: Required columns
    if "individuals" not in clean_df.columns:
        raise ValueError("Missing required column: 'individuals'")
    if "baseline" not in clean_df.columns:
        raise ValueError("Missing required column: 'baseline'")

    # Drop 'desired' column if present (used for test evaluation only)
    if "desired" in clean_df.columns:
        clean_df = clean_df.drop(columns=["desired"])

    # Optional priority column (if provided, use it to order individuals)
    custom_priority_order = None
    if "priority" in clean_df.columns:
        custom_priority_order = (
            clean_df.sort_values(by="priority")["individuals"]
            .astype(str)
            .tolist()
        )
        clean_df = clean_df.drop(columns=["priority"])

    # Check 4: Check for null/empty/whitespace in individuals and baseline
    for col in ["individuals", "baseline"]:
        if clean_df[col].isna().any():
            raise ValueError(f"Column '{col}' contains null/missing values.")
        clean_df[col] = clean_df[col].astype(str).str.strip()
        if (clean_df[col] == "").any():
            raise ValueError(f"Column '{col}' contains empty or whitespace-only values.")

    # Check 3: Unique individuals
    if clean_df["individuals"].duplicated().any():
        dupes = clean_df["individuals"][clean_df["individuals"].duplicated()].tolist()
        raise ValueError(f"Duplicate individuals found: {dupes}")

    # Identify class preference columns
    class_columns = [
        col for col in clean_df.columns if col not in ("individuals", "baseline")
    ]
    if not class_columns:
        raise ValueError(
            "No class preference columns found. Expected at least one class column."
        )

    # Check 2: Exact set equality between class columns and baseline classes
    baseline_classes = set(clean_df["baseline"])
    class_col_set = set(class_columns)
    if class_col_set != baseline_classes:
        missing_in_baseline = class_col_set - baseline_classes
        missing_in_cols = baseline_classes - class_col_set
        err_parts = []
        if missing_in_baseline:
            err_parts.append(
                f"classes in columns but not in baseline: {sorted(missing_in_baseline)}"
            )
        if missing_in_cols:
            err_parts.append(
                f"classes in baseline but not in columns: {sorted(missing_in_cols)}"
            )
        raise ValueError(
            f"Class preference columns and baseline classes must match exactly. {'; '.join(err_parts)}"
        )

    # Check 5 & Check 6: Validate ranks and fill blanks
    for idx, row in clean_df.iterrows():
        ind = row["individuals"]
        provided_ranks: Dict[str, int] = {}

        for c in class_columns:
            val = row[c]
            # Explicitly reject boolean types (which are subclasses of int in Python)
            if isinstance(val, (bool, np.bool_)):
                raise ValueError(
                    f"Invalid rank '{val}' for individual '{ind}' in class '{c}'. "
                    "Ranks must be positive integers, not boolean values."
                )

            # Check if missing / blank
            if pd.isna(val) or val is None:
                continue
            if isinstance(val, str):
                val_str = val.strip()
                if val_str == "" or val_str.lower() in ("nan", "none", "null", "na", "n/a", "-"):
                    continue
                try:
                    num_val = float(val_str)
                except ValueError:
                    raise ValueError(
                        f"Invalid rank '{val}' for individual '{ind}' in class '{c}'. "
                        "Ranks must be positive integers."
                    )
            elif isinstance(val, (int, float, np.integer, np.floating)):
                num_val = float(val)
            else:
                raise ValueError(
                    f"Invalid rank type for individual '{ind}' in class '{c}': {type(val)}"
                )

            if not num_val.is_integer() or num_val <= 0:
                raise ValueError(
                    f"Invalid rank {val} for individual '{ind}' in class '{c}'. "
                    "Ranks must be positive integers {1, 2, ...}."
                )
            provided_ranks[c] = int(num_val)

        # Validate contiguity of provided ranks: {1, 2, ..., n}
        if provided_ranks:
            unique_ranks = set(provided_ranks.values())
            max_rank = max(unique_ranks)
            expected_ranks = set(range(1, max_rank + 1))
            if unique_ranks != expected_ranks:
                missing_ranks = sorted(expected_ranks - unique_ranks)
                raise ValueError(
                    f"Individual '{ind}' has gap in ranks: provided {sorted(unique_ranks)}, "
                    f"missing intermediate ranks {missing_ranks}. Expected contiguous set {list(range(1, max_rank + 1))}."
                )
            fill_rank = max_rank + 1
        else:
            # All classes blank: indifferent among all classes
            fill_rank = 1

        # Fill missing / blank class cells
        for c in class_columns:
            if c not in provided_ranks:
                clean_df.at[idx, c] = fill_rank
            else:
                clean_df.at[idx, c] = provided_ranks[c]

    # Convert class columns to integer type
    for c in class_columns:
        clean_df[c] = clean_df[c].astype(int)

    # Check 7: Tie-breaking priority order
    if custom_priority_order is not None:
        priority_order = custom_priority_order
    else:
        priority_order = compute_priority_order(clean_df["individuals"].tolist())

    return clean_df, class_columns, priority_order


def find_pareto_improving_cycle(
    df: pd.DataFrame,
    class_columns: List[str],
) -> Optional[List[str]]:
    """Detect if any Pareto-improving trading cycles exist among assigned classes.

    Constructs a directed trade graph where a directed edge exists from student i to student j
    if student i weakly prefers student j's assigned class over their own assigned class
    (rank_i(assigned_j) <= rank_i(assigned_i)).
    The edge is 'strict' if student i strictly prefers student j's class
    (rank_i(assigned_j) < rank_i(assigned_i)).

    A Pareto-improving trade cycle exists if and only if there is a directed cycle
    in this graph containing at least one strict edge.

    Returns:
        List of student IDs forming the cycle if found, or None if Pareto efficient.
    """
    individuals = df["individuals"].astype(str).tolist()
    assigned = dict(zip(individuals, df["assigned"].astype(str)))

    # Prefs: student -> class -> rank
    prefs: Dict[str, Dict[str, int]] = {}
    for _, row in df.iterrows():
        s = str(row["individuals"])
        prefs[s] = {c: int(row[c]) for c in class_columns}

    # Build adjacency list: node -> list of (neighbor, is_strict)
    adj: Dict[str, List[Tuple[str, bool]]] = {s: [] for s in individuals}
    strict_edges: List[Tuple[str, str]] = []

    for u in individuals:
        curr_class = assigned[u]
        curr_rank = prefs[u][curr_class]
        for v in individuals:
            if u == v:
                continue
            cand_class = assigned[v]
            cand_rank = prefs[u][cand_class]
            if cand_rank < curr_rank:
                adj[u].append((v, True))
                strict_edges.append((u, v))
            elif cand_rank == curr_rank and cand_class != curr_class:
                adj[u].append((v, False))

    if not strict_edges:
        return None

    # For each strict edge (u, v), check if u is reachable from v via allowed edges
    for start, next_node in strict_edges:
        queue = [next_node]
        visited: Set[str] = {next_node}
        parent: Dict[str, str] = {next_node: start}
        found = False

        while queue:
            curr = queue.pop(0)
            if curr == start:
                found = True
                break
            for nxt, _ in adj[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = curr
                    queue.append(nxt)

        if found:
            # Reconstruct cycle
            cycle = [start]
            curr = parent[start]
            while curr != start:
                cycle.append(curr)
                curr = parent[curr]
            cycle.reverse()
            return cycle

    return None


def validate_post_trade(
    input_df: pd.DataFrame,
    results_df: pd.DataFrame,
    class_columns: List[str],
) -> None:
    """Validate trading outcome according to Post-Trade sanity checks.

    Checks:
    1. Capacity conservation: For every course c, count_final(c) == count_baseline(c).
    2. Individual Rationality (IR): For every student i, assigned_rank_i <= baseline_rank_i.
    3. Completeness: Every student in 'individuals' is assigned to exactly one class.
    4. Pareto efficiency verification: Verify no Pareto-improving trading cycles exist.
    """
    # Check 3: Completeness
    if results_df["individuals"].isna().any() or (results_df["individuals"].astype(str).str.strip() == "").any():
        raise ValueError("Completeness violated: null or empty individual names in results.")

    if len(results_df) != len(input_df) or results_df["individuals"].duplicated().any():
        dupes = results_df["individuals"][results_df["individuals"].duplicated()].tolist()
        raise ValueError(
            f"Completeness violated: duplicate student assignments found in results: {dupes}"
        )

    input_students = set(input_df["individuals"].astype(str))
    result_students = set(results_df["individuals"].astype(str))
    if input_students != result_students:
        missing = input_students - result_students
        extra = result_students - input_students
        raise ValueError(
            f"Completeness violated. Missing students: {missing}, Extra students: {extra}"
        )

    if results_df["assigned"].isna().any():
        raise ValueError("Completeness violated: null assignments found in results.")

    if (results_df["assigned"].astype(str).str.strip() == "").any():
        raise ValueError("Completeness violated: empty assignments found in results.")

    # Check baseline integrity against input_df
    input_baseline = dict(
        zip(input_df["individuals"].astype(str), input_df["baseline"].astype(str))
    )
    for _, row in results_df.iterrows():
        s = str(row["individuals"])
        if s in input_baseline and str(row["baseline"]) != input_baseline[s]:
            raise ValueError(
                f"Baseline integrity violated for individual '{s}': "
                f"expected baseline '{input_baseline[s]}', but results record '{row['baseline']}'."
            )

    # Check 1: Capacity conservation
    baseline_counts = Counter(input_df["baseline"].astype(str))
    final_counts = Counter(results_df["assigned"].astype(str))
    for c in class_columns:
        if baseline_counts[c] != final_counts[c]:
            raise ValueError(
                f"Capacity conservation violated for class '{c}': "
                f"baseline has {baseline_counts[c]} slots, final allocation has {final_counts[c]} slots."
            )

    # Build rank lookup
    prefs: Dict[str, Dict[str, int]] = {}
    for _, row in input_df.iterrows():
        s = str(row["individuals"])
        prefs[s] = {c: int(row[c]) for c in class_columns}

    # Check 2: Individual Rationality (IR)
    for _, row in results_df.iterrows():
        s = str(row["individuals"])
        baseline_class = str(row["baseline"])
        assigned_class = str(row["assigned"])

        if assigned_class not in prefs[s]:
            raise ValueError(
                f"Assigned class '{assigned_class}' for individual '{s}' is not a valid class."
            )

        baseline_rank = prefs[s][baseline_class]
        assigned_rank = prefs[s][assigned_class]

        if assigned_rank > baseline_rank:
            raise ValueError(
                f"Individual Rationality violated for '{s}': "
                f"assigned {assigned_class} (rank {assigned_rank}) is strictly worse than "
                f"baseline {baseline_class} (rank {baseline_rank})."
            )

    # Check 4: Pareto efficiency verification
    merged_df = results_df[["individuals", "baseline", "assigned"]].copy()
    for c in class_columns:
        merged_df[c] = merged_df["individuals"].map(lambda s: prefs[s][c])

    improving_cycle = find_pareto_improving_cycle(merged_df, class_columns)
    if improving_cycle is not None:
        raise ValueError(
            f"Pareto efficiency violated. Pareto-improving trading cycle detected: "
            f"{' -> '.join(improving_cycle)} -> {improving_cycle[0]}"
        )
