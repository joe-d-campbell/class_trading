"""Runner for the TA Class Trading System.

Integrates input validation, the Top Trading Cycles mechanism for weak preferences,
post-trade sanity checks, and generates enriched allocation reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Union

import pandas as pd

# Handle imports whether package is imported or run as a script
try:
    from .algorithm import top_trading_cycles_weak_pref
    from .validator import validate_post_trade, validate_pre_trade
except ImportError:
    from algorithm import top_trading_cycles_weak_pref
    from validator import validate_post_trade, validate_pre_trade


def run_trading(
    input_csv: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Execute the TA class trading pipeline on an input CSV.

    Args:
        input_csv: Path to input CSV file.
        output_csv: Optional path to save the enriched output CSV file.

    Returns:
        enriched_df: DataFrame with columns:
            - individuals: student ID / name
            - baseline: baseline assigned class
            - assigned: final assigned class
            - baseline_rank: rank of baseline class for the student
            - assigned_rank: rank of final assigned class for the student
            - improved: boolean indicating whether assigned rank < baseline rank
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_df = pd.read_csv(input_path)

    # 1. Pre-trade validation
    validated_df, class_columns, priority_order = validate_pre_trade(raw_df)

    # 2. Execute trading algorithm
    assignments = top_trading_cycles_weak_pref(
        validated_df,
        class_columns=class_columns,
        priority_order=priority_order,
    )

    # 3. Build enriched results DataFrame
    records = []
    for _, row in validated_df.iterrows():
        s = str(row["individuals"])
        base_class = str(row["baseline"])
        asgn_class = assignments[s]
        base_rank = int(row[base_class])
        asgn_rank = int(row[asgn_class])
        records.append(
            {
                "individuals": s,
                "baseline": base_class,
                "assigned": asgn_class,
                "baseline_rank": base_rank,
                "assigned_rank": asgn_rank,
                "improved": asgn_rank < base_rank,
            }
        )

    enriched_df = pd.DataFrame(records)

    # 4. Post-trade validation
    validate_post_trade(validated_df, enriched_df, class_columns)

    # 5. Save output if requested
    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        enriched_df.to_csv(out_path, index=False)

    return enriched_df


def main() -> None:
    """CLI entry point for running the TA class trading system."""
    parser = argparse.ArgumentParser(
        description="TA Class Trading System (Jaramillo & Manjunath 2012 Top Trading Cycles)"
    )
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to output CSV file",
        default=None,
    )

    args = parser.parse_args()

    try:
        results = run_trading(args.input_csv, args.output)
        print("=== TA Class Trading Final Allocation ===")
        print(results.to_string(index=False))
        if args.output:
            print(f"\nAllocation saved to: {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
