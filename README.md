# TA Class Trading System

An implementation of the **Top Trading Cycles mechanism for weak preferences** ($TC^\prec$) developed by **Paula Jaramillo and Vikram Manjunath (2012)** (*Journal of Economic Theory*). 

This package is designed for university departments to reallocate Teaching Assistant (TA) course assignments among students. Students submit their preference rankings over courses, and the system executes mutually beneficial reallocations subject to hard course capacity constraints.

---

## Key Economic Guarantees

* **Individual Rationality (IR)**: No student is ever assigned a course worse than their initial baseline course.
* **Pareto Efficiency**: The final allocation admits no trading cycles that could strictly improve any student without making another worse off.
* **Capacity Conservation**: The exact number of TA seats allocated to each class remains unchanged before and after trading.
* **Support for Weak Preferences (Indifferences)**: Students can rank multiple classes as equally desirable (e.g. tying multiple courses at rank 1). Ties are broken using a fixed priority order $\prec$.

---

## Project Structure

```
class_trading/
├── code/
│   ├── algorithm.py     # Core Jaramillo & Manjunath (2012) TC^≺ implementation
│   ├── validator.py     # Comprehensive pre-trade and post-trade sanity checks
│   └── runner.py        # CLI entry point and high-level Python API
└── tests/
    ├── test_trading.py  # Automated test suite (27 unit and integration tests)
    ├── test_1.csv       # Benchmark market 1 (jc, am, nl, mn)
    ├── test_2.csv       # Benchmark market 2
    ├── paper_example_1_prio_2_3.csv     # Paper Example 1 (priority 2 ≺ 3)
    ├── paper_example_1_prio_3_2.csv     # Paper Example 1 (priority 3 ≺ 2)
    ├── paper_example_2.csv              # Paper Example 2 (Section 5, 11 agents)
    ├── paper_example_3_profile_R.csv    # Paper Example 3 (Profile R)
    ├── paper_example_3_profile_Rprime.csv # Paper Example 3 (Profile R')
    ├── test_multi_slot.csv              # Multi-capacity class test market
    ├── test_all_satisfied.csv           # Zero-trade baseline test market
    └── test_disjoint_cycles.csv         # Disjoint independent trading cycles
```

---

## Installation & Requirements

* Python 3.9+
* Required libraries: `pandas`, `pytest`

To install dependencies:

```bash
pip install pandas pytest
```

---

## Walkthrough Example (`jc`, `am`, `nl`, `mn`)

To illustrate the mechanism, consider a 4-student department with classes `micro` (1 slot), `macro` (1 slot), and `metrics` (2 slots).

### Input CSV (`market.csv`)

```csv
individuals,baseline,micro,macro,metrics
jc,micro,1,2,1
am,metrics,2,2,1
nl,macro,2,1,2
mn,metrics,1,3,2
```

In this market:
* `nl` is baselined in `macro` and ranks `macro` first ($1$).
* `am` is baselined in `metrics` and ranks `metrics` first ($1$).
* `jc` is baselined in `micro` and is indifferent between `micro` and `metrics` (both rank $1$).
* `mn` is baselined in `metrics` (rank $2$), but strictly prefers `micro` (rank $1$).

### Running the Trade

```bash
python3 code/runner.py market.csv
```

### Output Table

| `individuals` | `baseline` | `assigned` | `baseline_rank` | `assigned_rank` | `improved` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **jc** | micro | metrics | 1 | 1 | `False` |
| **am** | metrics | metrics | 1 | 1 | `False` |
| **nl** | macro | macro | 1 | 1 | `False` |
| **mn** | metrics | micro | 2 | 1 | `True` |

### Intuition:
1. `nl` and `am` already hold their top-ranked classes and immediately depart without trading.
2. `jc` (holding `micro`) and `mn` (holding `metrics`) form a mutually beneficial trade: `mn` receives `micro` (improving from rank $2$ to rank $1$), while `jc` receives `metrics` (maintaining their top rank $1$).
3. **Result**: Every student receives their rank-1 course, capacity is exactly conserved (`micro: 1, macro: 1, metrics: 2`), and no student is worse off.

---

## Input CSV Specification

Input CSVs must follow these rules:

1. **Required Columns**:
   * `individuals`: Unique identifier or name for each student.
   * `baseline`: The student's initially assigned course.
2. **Class Columns**:
   * Exactly one column for every distinct course appearing in `baseline`.
   * Column headers must match the values in `baseline`.
3. **Preference Values (Ranks)**:
   * Positive integers ($1, 2, \dots, n$) where **$1$ is most preferred**.
   * Weak preferences (ties) are supported: students may assign the same rank to multiple courses.
   * Ranks provided by each student must form a contiguous sequence from $1$ to $n$ (no skipped numbers).
   * Blank/empty cells are automatically assigned rank $n+1$ (least preferred).
4. **Optional Columns**:
   * `priority`: Positive integer defining tie-breaking order ($1$ is highest priority). If omitted, a fair tie-breaking order is generated deterministically using SHA-256 hashes of student names.
   * `desired`: Target allocation column (useful for test benchmarks; automatically stripped during trading).

---

## Output Columns

The output provides an enriched summary table:

| Column | Type | Description |
| :--- | :--- | :--- |
| `individuals` | string | Student identifier |
| `baseline` | string | Initially assigned course |
| `assigned` | string | Final course allocation after trade |
| `baseline_rank` | integer | Rank of the baseline course according to student's preferences |
| `assigned_rank` | integer | Rank of the assigned course according to student's preferences |
| `improved` | boolean | `True` if student obtained a strictly better rank (`assigned_rank < baseline_rank`), else `False` |

---

## Automated Validation & Sanity Checks

Every run automatically executes rigorous pre- and post-trade validations:

### Pre-Trade Checks (`validate_pre_trade`)
1. Verifies presence of `individuals` and `baseline`.
2. Asserts exact set equality: $\text{set}(\text{class columns}) == \text{set}(\text{baseline courses})$.
3. Rejects duplicate entries in `individuals`.
4. Rejects null, empty, or whitespace-only entries.
5. Verifies rank contiguity $\{1, \dots, n\}$ for each student and rejects non-integer/boolean ranks.
6. Imputes unranked/blank cells with $n+1$.
7. Establishes deterministic priority ordering.

### Post-Trade Checks (`validate_post_trade`)
1. **Capacity Conservation**: Total slots allocated to each course match baseline capacity.
2. **Individual Rationality**: $\text{assigned\_rank}_i \le \text{baseline\_rank}_i$ for every student $i$.
3. **Completeness**: Every student receives exactly one valid course; no one is dropped or duplicated.
4. **Pareto Optimality**: Performs depth-first cycle search on the preference improvement graph to guarantee no improving trading cycles exist.

---

## Running Tests

Run the complete test suite:

```bash
pytest -v tests/test_trading.py
```

All 27 tests (covering baseline matches, pre/post sanity checks, paper examples, multi-capacity slots, and CLI execution) should pass.

---

## References

* **Jaramillo, P., & Manjunath, V. (2012)**. *The difference indifference makes in strategy-proof allocation of objects*. **Journal of Economic Theory**, 147(5), 1913–1946.
* **Shapley, L., & Scarf, H. (1974)**. *On cores and indivisibility*. **Journal of Mathematical Economics**, 1(1), 23–37.
