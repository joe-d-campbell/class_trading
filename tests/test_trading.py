"""Comprehensive test suite for the TA Class Trading System.

Tests cover:
1. Exact replication of test_1.csv and test_2.csv matching 'desired'.
2. Pre-trade sanity checks:
   - Required columns
   - Exact set equality of class columns and baseline classes
   - Unique individuals
   - No null, empty, or whitespace values
   - Rank contiguity and gap detection
   - Blank/missing cell filling with n+1
   - SHA-256 deterministic priority ordering
3. Post-trade sanity checks:
   - Capacity conservation
   - Individual rationality (IR)
   - Completeness
   - Pareto efficiency verification (improving trading cycle detection)
4. Theoretical edge cases:
   - Example 2 from Jaramillo & Manjunath (2012)
   - Multi-slot capacity trading
   - All-satisfied baseline (no trade needed)
   - Disjoint cycles
5. CLI execution and output generation.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Dict


import numpy as np
import pandas as pd
import pytest

# Ensure code directory is in sys.path (standard library also has a 'code' module)
CODE_DIR = Path(__file__).resolve().parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from algorithm import top_trading_cycles_weak_pref
from runner import run_trading
from validator import (
    compute_priority_order,
    find_pareto_improving_cycle,
    validate_post_trade,
    validate_pre_trade,
)



TESTS_DIR = Path(__file__).parent


# =====================================================================
# 1. Baseline Test Cases (test_1.csv and test_2.csv)
# =====================================================================

def test_run_test_1_matches_desired():
    """Verify that running the algorithm on test_1.csv exactly matches 'desired'."""
    csv_path = TESTS_DIR / "test_1.csv"
    df = pd.read_csv(csv_path)
    expected_desired = dict(zip(df["individuals"], df["desired"]))

    enriched_df = run_trading(csv_path)
    actual_assigned = dict(zip(enriched_df["individuals"], enriched_df["assigned"]))

    assert actual_assigned == expected_desired, (
        f"test_1.csv output mismatch:\nExpected: {expected_desired}\nActual: {actual_assigned}"
    )


def test_run_test_2_matches_desired():
    """Verify that running the algorithm on test_2.csv exactly matches 'desired'."""
    csv_path = TESTS_DIR / "test_2.csv"
    df = pd.read_csv(csv_path)
    expected_desired = dict(zip(df["individuals"], df["desired"]))

    enriched_df = run_trading(csv_path)
    actual_assigned = dict(zip(enriched_df["individuals"], enriched_df["assigned"]))

    assert actual_assigned == expected_desired, (
        f"test_2.csv output mismatch:\nExpected: {expected_desired}\nActual: {actual_assigned}"
    )


def test_enriched_dataframe_columns_and_types():
    """Verify that runner outputs all required enriched columns with correct semantics."""
    csv_path = TESTS_DIR / "test_1.csv"
    enriched_df = run_trading(csv_path)

    expected_cols = [
        "individuals",
        "baseline",
        "assigned",
        "baseline_rank",
        "assigned_rank",
        "improved",
    ]
    assert list(enriched_df.columns) == expected_cols

    for _, row in enriched_df.iterrows():
        assert isinstance(row["improved"], (bool, np.bool_))
        assert row["improved"] == (row["assigned_rank"] < row["baseline_rank"])
        assert row["assigned_rank"] <= row["baseline_rank"]  # IR guarantee


# =====================================================================
# 2. Pre-Trade Sanity Checks
# =====================================================================

def test_pre_trade_missing_required_columns():
    """Check 1: Raise ValueError when 'individuals' or 'baseline' is missing."""
    df_missing_ind = pd.DataFrame({
        "baseline": ["micro", "macro"],
        "micro": [1, 2],
        "macro": [2, 1],
    })
    with pytest.raises(ValueError, match="Missing required column: 'individuals'"):
        validate_pre_trade(df_missing_ind)

    df_missing_base = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "micro": [1, 2],
        "macro": [2, 1],
    })
    with pytest.raises(ValueError, match="Missing required column: 'baseline'"):
        validate_pre_trade(df_missing_base)


def test_pre_trade_no_class_columns():
    """Check 1: Raise ValueError when no class preference columns exist."""
    df_no_classes = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
    })
    with pytest.raises(ValueError, match="No class preference columns found"):
        validate_pre_trade(df_no_classes)


def test_pre_trade_class_columns_baseline_mismatch():
    """Check 2: Raise ValueError when set(class_columns) != set(baseline)."""
    # Column 'metrics' exists, but no student is baselined in 'metrics'
    df_mismatch_1 = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "micro": [1, 2],
        "macro": [2, 1],
        "metrics": [3, 3],
    })
    with pytest.raises(ValueError, match="Class preference columns and baseline classes must match exactly"):
        validate_pre_trade(df_mismatch_1)

    # Baseline has 'metrics', but column 'metrics' is missing
    df_mismatch_2 = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "metrics"],
        "micro": [1, 2],
        "macro": [2, 1],
    })
    with pytest.raises(ValueError, match="Class preference columns and baseline classes must match exactly"):
        validate_pre_trade(df_mismatch_2)


def test_pre_trade_duplicate_individuals():
    """Check 3: Raise ValueError when duplicate names/IDs appear in 'individuals'."""
    df_dupes = pd.DataFrame({
        "individuals": ["alice", "alice"],
        "baseline": ["micro", "macro"],
        "micro": [1, 2],
        "macro": [2, 1],
    })
    with pytest.raises(ValueError, match="Duplicate individuals found"):
        validate_pre_trade(df_dupes)


def test_pre_trade_null_empty_whitespace_values():
    """Check 4: Raise ValueError when 'individuals' or 'baseline' contains null, empty, or whitespace values."""
    # Null in individuals
    df_null_ind = pd.DataFrame({
        "individuals": ["alice", None],
        "baseline": ["micro", "macro"],
        "micro": [1, 2],
        "macro": [2, 1],
    })
    with pytest.raises(ValueError, match="Column 'individuals' contains null/missing values"):
        validate_pre_trade(df_null_ind)

    # Whitespace in individuals
    df_ws_ind = pd.DataFrame({
        "individuals": ["alice", "   "],
        "baseline": ["micro", "macro"],
        "micro": [1, 2],
        "macro": [2, 1],
    })
    with pytest.raises(ValueError, match="Column 'individuals' contains empty or whitespace-only values"):
        validate_pre_trade(df_ws_ind)

    # Whitespace in baseline
    df_ws_base = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "  "],
        "micro": [1, 2],
        "macro": [2, 1],
    })
    with pytest.raises(ValueError, match="Column 'baseline' contains empty or whitespace-only values"):
        validate_pre_trade(df_ws_base)


def test_pre_trade_rank_gaps_raise_value_error():
    """Check 5: Raise ValueError when ranks have gaps (e.g. 1 and 3 with no 2)."""
    # Gap: {1, 3} missing 2
    df_gap = pd.DataFrame({
        "individuals": ["alice", "bob", "carol"],
        "baseline": ["micro", "macro", "metrics"],
        "micro": [1, 1, 2],
        "macro": [3, 2, 1],  # alice has 1 and 3, missing 2
        "metrics": [1, 3, 3],
    })
    with pytest.raises(ValueError, match="gap in ranks"):
        validate_pre_trade(df_gap)

    # Starting at 2 instead of 1: {2, 3} missing 1
    df_no_one = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "micro": [2, 1],
        "macro": [3, 2],  # alice has 2 and 3, missing 1
    })
    with pytest.raises(ValueError, match="gap in ranks"):
        validate_pre_trade(df_no_one)


def test_pre_trade_invalid_ranks():
    """Check 5: Raise ValueError for non-positive or non-integer rank values."""
    # Rank 0
    df_zero = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "micro": [0, 1],
        "macro": [1, 2],
    })
    with pytest.raises(ValueError, match="Invalid rank"):
        validate_pre_trade(df_zero)

    # Negative rank
    df_neg = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "micro": [-1, 1],
        "macro": [1, 2],
    })
    with pytest.raises(ValueError, match="Invalid rank"):
        validate_pre_trade(df_neg)

    # Fractional rank
    df_float = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "micro": [1.5, 1],
        "macro": [2, 2],
    })
    with pytest.raises(ValueError, match="Invalid rank"):
        validate_pre_trade(df_float)

    # Non-numeric string rank
    df_str = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "micro": ["first", 1],
        "macro": [2, 2],
    })
    with pytest.raises(ValueError, match="Invalid rank"):
        validate_pre_trade(df_str)


def test_pre_trade_fill_blank_and_missing_cells():
    """Check 6: Blank or missing class cells are filled with n+1."""
    df_with_blanks = pd.DataFrame({
        "individuals": ["alice", "bob", "carol"],
        "baseline": ["micro", "macro", "metrics"],
        "micro": [1, 2, None],        # carol missing micro -> filled with 2
        "macro": [2, None, 1],        # bob missing macro (baseline) -> filled with 3
        "metrics": [np.nan, 1, ""],    # alice missing metrics -> filled with 3; carol blank metrics -> filled with 2
    })

    clean_df, class_cols, _ = validate_pre_trade(df_with_blanks)

    # alice provided {1, 2} (max=2), so metrics should be filled with 2+1 = 3
    assert clean_df.loc[clean_df["individuals"] == "alice", "metrics"].values[0] == 3
    # bob provided {2, 1} (max=2), so macro should be filled with 2+1 = 3
    assert clean_df.loc[clean_df["individuals"] == "bob", "macro"].values[0] == 3
    # carol provided {1} for macro (max=1), so micro and metrics should be filled with 1+1 = 2
    assert clean_df.loc[clean_df["individuals"] == "carol", "micro"].values[0] == 2
    assert clean_df.loc[clean_df["individuals"] == "carol", "metrics"].values[0] == 2


def test_pre_trade_all_blank_cells():
    """Check 6: When all class cells are blank for an individual, fill with 1 (indifferent)."""
    df_all_blank = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "micro": [None, 1],
        "macro": [np.nan, 2],
    })
    clean_df, _, _ = validate_pre_trade(df_all_blank)
    assert clean_df.loc[clean_df["individuals"] == "alice", "micro"].values[0] == 1
    assert clean_df.loc[clean_df["individuals"] == "alice", "macro"].values[0] == 1


def test_pre_trade_sha256_priority_order():
    """Check 7: Priority order ≺ is derived deterministically by sorting sha256 hex digest."""
    students = ["jc", "am", "nl", "mn"]
    expected_order = sorted(
        students,
        key=lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest(),
    )
    # Expected order: nl (1843...), jc (9f1c...), am (ab6d...), mn (ea43...)
    assert expected_order == ["nl", "jc", "am", "mn"]
    assert compute_priority_order(students) == expected_order


# =====================================================================
# 3. Post-Trade Sanity Checks
# =====================================================================

def test_post_trade_capacity_violation():
    """Check 1: Raise ValueError when course capacity is not conserved."""
    input_df = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "micro": [1, 2],
        "macro": [2, 1],
    })
    # Both assigned to micro (capacity violated: micro=2, macro=0)
    invalid_results = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "assigned": ["micro", "micro"],
        "baseline_rank": [1, 1],
        "assigned_rank": [1, 2],
        "improved": [False, False],
    })
    with pytest.raises(ValueError, match="Capacity conservation violated for class"):
        validate_post_trade(input_df, invalid_results, ["micro", "macro"])



def test_post_trade_ir_violation():
    """Check 2: Raise ValueError when an individual receives a rank worse than baseline."""
    input_df = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "micro": [1, 1],
        "macro": [2, 2],
    })
    # alice had micro (rank 1), now assigned macro (rank 2) -> IR violation!
    invalid_results = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "assigned": ["macro", "micro"],
        "baseline_rank": [1, 2],
        "assigned_rank": [2, 1],
        "improved": [False, True],
    })
    with pytest.raises(ValueError, match="Individual Rationality violated for 'alice'"):
        validate_post_trade(input_df, invalid_results, ["micro", "macro"])


def test_post_trade_completeness_violation():
    """Check 3: Raise ValueError if students are missing, extra, or assigned null/empty."""
    input_df = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "micro": [1, 2],
        "macro": [2, 1],
    })
    # Missing bob
    missing_results = pd.DataFrame({
        "individuals": ["alice"],
        "baseline": ["micro"],
        "assigned": ["macro"],
        "baseline_rank": [1],
        "assigned_rank": [2],
        "improved": [False],
    })
    with pytest.raises(ValueError, match="Completeness violated"):
        validate_post_trade(input_df, missing_results, ["micro", "macro"])

    # Null assigned
    null_results = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "assigned": ["micro", None],
        "baseline_rank": [1, 1],
        "assigned_rank": [1, 1],
        "improved": [False, False],
    })
    with pytest.raises(ValueError, match="Completeness violated: null assignments found"):
        validate_post_trade(input_df, null_results, ["micro", "macro"])


def test_post_trade_pareto_efficiency_violation():
    """Check 4: Raise ValueError when an allocation has a Pareto-improving trading cycle."""
    # alice and bob both prefer each other's baseline class:
    # alice (baseline micro): ranks macro: 1, micro: 2
    # bob (baseline macro): ranks micro: 1, macro: 2
    input_df = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "micro": [2, 1],
        "macro": [1, 2],
    })
    # Suppose we falsely kept them at their baseline allocations:
    # alice gets micro (rank 2), bob gets macro (rank 2)
    # Both are individually rational (rank 2 == baseline rank 2),
    # and capacity is conserved (micro: 1, macro: 1).
    # BUT trading micro for macro is Pareto improving!
    inefficient_results = pd.DataFrame({
        "individuals": ["alice", "bob"],
        "baseline": ["micro", "macro"],
        "assigned": ["micro", "macro"],
        "baseline_rank": [2, 2],
        "assigned_rank": [2, 2],
        "improved": [False, False],
    })
    with pytest.raises(ValueError, match="Pareto efficiency violated"):
        validate_post_trade(input_df, inefficient_results, ["micro", "macro"])


def test_find_pareto_improving_cycle_three_students():
    """Verify Pareto-improving cycle detection on a 3-way cycle with weak preferences."""
    # alice (has A): wants B (strict, rank 1 < 2)
    # bob (has B): indifferent between B and C (rank 1 == 1)
    # carol (has C): wants A (strict, rank 1 < 2)
    df = pd.DataFrame({
        "individuals": ["alice", "bob", "carol"],
        "baseline": ["A", "B", "C"],
        "assigned": ["A", "B", "C"],
        "A": [2, 3, 1],
        "B": [1, 1, 3],
        "C": [3, 1, 2],
    })
    cycle = find_pareto_improving_cycle(df, ["A", "B", "C"])
    assert cycle is not None
    assert set(cycle) == {"alice", "bob", "carol"}


# =====================================================================
# 4. Advanced Algorithm & Economic Theory Tests
# =====================================================================

def test_paper_example_1_prio_2_3():
    """Verify Example 1 from Jaramillo & Manjunath (2012) with priority 2 < 3 loaded from CSV."""
    csv_path = TESTS_DIR / "paper_example_1_prio_2_3.csv"
    df = pd.read_csv(csv_path)
    expected_desired = dict(zip(df["individuals"].astype(str), df["desired"].astype(str)))

    res = run_trading(csv_path)
    actual_assigned = dict(zip(res["individuals"].astype(str), res["assigned"].astype(str)))
    assert actual_assigned == expected_desired, f"Example 1 (2 < 3) mismatch: {actual_assigned} != {expected_desired}"


def test_paper_example_1_prio_3_2():
    """Verify Example 1 from Jaramillo & Manjunath (2012) with priority 3 < 2 loaded from CSV."""
    csv_path = TESTS_DIR / "paper_example_1_prio_3_2.csv"
    df = pd.read_csv(csv_path)
    expected_desired = dict(zip(df["individuals"].astype(str), df["desired"].astype(str)))

    res = run_trading(csv_path)
    actual_assigned = dict(zip(res["individuals"].astype(str), res["assigned"].astype(str)))
    assert actual_assigned == expected_desired, f"Example 1 (3 < 2) mismatch: {actual_assigned} != {expected_desired}"


def test_jaramillo_manjunath_example_2():
    """Test the complete 11-agent market from Jaramillo & Manjunath (2012) Example 2 loaded from CSV."""
    csv_path = TESTS_DIR / "paper_example_2.csv"
    df = pd.read_csv(csv_path)
    expected_desired = dict(zip(df["individuals"].astype(str), df["desired"].astype(str)))

    res = run_trading(csv_path)
    actual_assigned = dict(zip(res["individuals"].astype(str), res["assigned"].astype(str)))
    assert actual_assigned == expected_desired, f"Example 2 mismatch: {actual_assigned} != {expected_desired}"


def test_paper_example_3_profile_R():
    """Verify Example 3 (Profile R) from Jaramillo & Manjunath (2012) loaded from CSV."""
    csv_path = TESTS_DIR / "paper_example_3_profile_R.csv"
    df = pd.read_csv(csv_path)
    expected_desired = dict(zip(df["individuals"].astype(str), df["desired"].astype(str)))

    res = run_trading(csv_path)
    actual_assigned = dict(zip(res["individuals"].astype(str), res["assigned"].astype(str)))
    assert actual_assigned == expected_desired, f"Example 3 (R) mismatch: {actual_assigned} != {expected_desired}"


def test_paper_example_3_profile_Rprime():
    """Verify Example 3 (Profile R') from Jaramillo & Manjunath (2012) loaded from CSV."""
    csv_path = TESTS_DIR / "paper_example_3_profile_Rprime.csv"
    df = pd.read_csv(csv_path)
    expected_desired = dict(zip(df["individuals"].astype(str), df["desired"].astype(str)))

    res = run_trading(csv_path)
    actual_assigned = dict(zip(res["individuals"].astype(str), res["assigned"].astype(str)))
    assert actual_assigned == expected_desired, f"Example 3 (R') mismatch: {actual_assigned} != {expected_desired}"


def test_multi_slot_capacity_class_trading():
    """Verify trading with multiple students assigned to the same class (multi-slot capacity) loaded from CSV."""
    csv_path = TESTS_DIR / "test_multi_slot.csv"
    df = pd.read_csv(csv_path)
    expected_desired = dict(zip(df["individuals"].astype(str), df["desired"].astype(str)))

    res = run_trading(csv_path)
    actual_assigned = dict(zip(res["individuals"].astype(str), res["assigned"].astype(str)))
    assert actual_assigned == expected_desired, f"Multi-slot mismatch: {actual_assigned} != {expected_desired}"
    assert Counter(res["assigned"]) == {"micro": 3, "macro": 2, "metrics": 2}


def test_all_satisfied_baseline_no_trades():
    """When everyone already has their rank 1 class, loaded from CSV."""
    csv_path = TESTS_DIR / "test_all_satisfied.csv"
    df = pd.read_csv(csv_path)
    expected_desired = dict(zip(df["individuals"].astype(str), df["desired"].astype(str)))

    res = run_trading(csv_path)
    actual_assigned = dict(zip(res["individuals"].astype(str), res["assigned"].astype(str)))
    assert actual_assigned == expected_desired, f"All satisfied mismatch: {actual_assigned} != {expected_desired}"


def test_two_disjoint_trading_cycles():
    """Verify that multiple independent trading cycles execute correctly loaded from CSV."""
    csv_path = TESTS_DIR / "test_disjoint_cycles.csv"
    df = pd.read_csv(csv_path)
    expected_desired = dict(zip(df["individuals"].astype(str), df["desired"].astype(str)))

    res = run_trading(csv_path)
    actual_assigned = dict(zip(res["individuals"].astype(str), res["assigned"].astype(str)))
    assert actual_assigned == expected_desired, f"Disjoint cycles mismatch: {actual_assigned} != {expected_desired}"


# =====================================================================
# 5. CLI Execution Tests
# =====================================================================

def test_cli_execution_with_output(tmp_path):
    """Verify that runner CLI runs cleanly and saves output to file."""
    output_file = tmp_path / "out_result.csv"
    cmd = [
        sys.executable,
        "code/runner.py",
        str(TESTS_DIR / "test_1.csv"),
        "--output",
        str(output_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed with error:\n{res.stderr}"
    assert "=== TA Class Trading Final Allocation ===" in res.stdout
    assert output_file.exists()

    saved_df = pd.read_csv(output_file)
    assert list(saved_df.columns) == [
        "individuals",
        "baseline",
        "assigned",
        "baseline_rank",
        "assigned_rank",
        "improved",
    ]
    assert len(saved_df) == 4
