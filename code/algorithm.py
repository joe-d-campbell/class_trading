"""Implementation of the Top Trading Cycles algorithm for weak preferences.

Reference:
Paula Jaramillo and Vikram Manjunath. "The difference indifference makes in
strategy-proof allocation of objects." Journal of Economic Theory 147 (2012) 1913-1946.
Section 5 and Appendix A.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


def top_trading_cycles_weak_pref(
    df: pd.DataFrame,
    class_columns: Optional[List[str]] = None,
    priority_order: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Execute the Top Trading Cycles algorithm for weak preferences with fixed tie-breaking (TC^≺).

    Args:
        df: Input DataFrame containing 'individuals', 'baseline', and class preference columns.
        class_columns: List of class columns. If None, derived from df.
        priority_order: List of student IDs ordered from highest priority (index 0)
            to lowest priority. If None, computed deterministically via SHA-256 hex digest.

    Returns:
        assigned: Dict mapping each student ID to their assigned class name.
    """
    if class_columns is None:
        class_columns = [
            c for c in df.columns if c not in ("individuals", "baseline", "desired")
        ]

    individuals: List[str] = df["individuals"].astype(str).tolist()

    if priority_order is None:
        priority_order = sorted(
            individuals,
            key=lambda s: hashlib.sha256(str(s).encode("utf-8")).hexdigest(),
        )
    priority_rank: Dict[str, int] = {s: i for i, s in enumerate(priority_order)}

    # Extract student preferences: student -> class -> integer rank (lower is better)
    prefs: Dict[str, Dict[str, int]] = {}
    for _, row in df.iterrows():
        s = str(row["individuals"])
        prefs[s] = {c: int(row[c]) for c in class_columns}

    # Expand each class c into k_c distinct slot items where k_c is the number of
    # students assigned to c in baseline.
    # Each student is endowed with one slot of their baseline class.
    # We identify each slot item initially by the student who holds it.
    item_class: Dict[str, str] = {
        str(row["individuals"]): str(row["baseline"])
        for _, row in df.iterrows()
    }

    # Holdings at step t: student -> item_id
    h: Dict[str, str] = {s: s for s in individuals}

    remaining_people: Set[str] = set(individuals)
    assigned: Dict[str, str] = {}

    step = 0
    max_steps = max(len(individuals) ** 2 + 10, 100)
    prev_p: Dict[str, str] = {}
    prev_h: Dict[str, str] = {}

    while remaining_people:
        step += 1
        if step > max_steps:
            raise RuntimeError(
                f"Algorithm exceeded maximum allowed steps ({max_steps}) without terminating."
            )

        # =====================================================================
        # 1. DEPARTURE PHASE (Appendix A)
        # =====================================================================
        while True:
            current_items = {h[p] for p in remaining_people}

            def get_top_items(person: str, items: Set[str]) -> Set[str]:
                """Return the subset of items that person ranks at the top among items."""
                avail_classes = {item_class[it] for it in items}
                best_rank = min(prefs[person][c] for c in avail_classes)
                return {
                    it
                    for it in items
                    if prefs[person][item_class[it]] == best_rank
                }

            # Candidate group G starts with all remaining people who are currently satisfied
            # (i) ht(i) in tau(Ri, ht(N_{t-1}))
            candidate_group: Set[str] = set()
            for p in remaining_people:
                top_items = get_top_items(p, current_items)
                if h[p] in top_items:
                    candidate_group.add(p)

            # Prune candidate_group until condition (ii) is met:
            # (ii) tau(Ri, ht(N_{t-1})) is a subset of ht(candidate_group)
            while True:
                cand_items = {h[p] for p in candidate_group}
                to_remove: Set[str] = set()
                for p in candidate_group:
                    top_items = get_top_items(p, current_items)
                    if not top_items.issubset(cand_items):
                        to_remove.add(p)
                if not to_remove:
                    break
                candidate_group -= to_remove

            if not candidate_group:
                # No more groups can depart in this step
                break

            # Candidate group departs!
            for p in candidate_group:
                assigned[p] = item_class[h[p]]
                remaining_people.remove(p)

        if not remaining_people:
            break

        # =====================================================================
        # 2. POINTING PHASE (Appendix A)
        # =====================================================================
        curr_items = {h[p] for p in remaining_people}

        def get_top_items_curr(person: str) -> Set[str]:
            avail_classes = {item_class[it] for it in curr_items}
            best_rank = min(prefs[person][c] for c in avail_classes)
            return {
                it
                for it in curr_items
                if prefs[person][item_class[it]] == best_rank
            }

        # St: satisfied people, Ut: unsatisfied people
        S_t: Set[str] = set()
        U_t: Set[str] = set()
        # Ci,t: people holding one of i's most preferred objects among remaining
        C: Dict[str, Set[str]] = {}
        for p in remaining_people:
            top_its = get_top_items_curr(p)
            if h[p] in top_its:
                S_t.add(p)
            else:
                U_t.add(p)
            C[p] = {other for other in remaining_people if h[other] in top_its}

        p_t: Dict[str, str] = {}
        unassigned: Set[str] = set(remaining_people)

        # Stage 1: Persistence
        # If t != 1, for each i whose pointee in t-1 has not departed and holds
        # the same object, i points at the same person.
        if step > 1:
            for p in list(unassigned):
                if p in prev_p:
                    old_target = prev_p[p]
                    if (
                        old_target in remaining_people
                        and h[old_target] == prev_h.get(old_target)
                    ):
                        p_t[p] = old_target
                        unassigned.remove(p)

        # Stage 2: Unique candidate pointee
        for p in list(unassigned):
            if len(C[p]) == 1:
                p_t[p] = next(iter(C[p]))
                unassigned.remove(p)

        # Stage 3: Unsatisfied candidate pointee
        for p in list(unassigned):
            unsat = C[p].intersection(U_t)
            if unsat:
                p_t[p] = min(unsat, key=lambda j: priority_rank[j])
                unassigned.remove(p)

        # Stages 4, 5, ...: Pointing through chains to Ut
        while unassigned:
            def get_path_to_ut(person: str) -> Tuple[Optional[int], Optional[str]]:
                """Find the distance and eventual person in Ut reached from person via p_t."""
                curr = person
                dist = 0
                visited: Set[str] = set()
                while curr in p_t:
                    if curr in visited:
                        return None, None
                    visited.add(curr)
                    nxt = p_t[curr]
                    dist += 1
                    if nxt in U_t:
                        return dist, nxt
                    curr = nxt
                return None, None

            # Find the minimum chain length to Ut among candidate pointees of unassigned people
            best_d: Optional[int] = None
            for p in unassigned:
                for cand in C[p]:
                    d, _ = get_path_to_ut(cand)
                    if d is not None:
                        if best_d is None or d < best_d:
                            best_d = d

            if best_d is None:
                # No unassigned person can reach Ut via current pointers
                break

            # Assign all unassigned people who have at least one candidate with distance best_d
            newly_assigned: Dict[str, str] = {}
            for p in unassigned:
                cands_at_best_d = []
                for cand in C[p]:
                    d, u = get_path_to_ut(cand)
                    if d == best_d and u is not None:
                        cands_at_best_d.append((cand, u))
                if cands_at_best_d:
                    # Sort candidates: primary key = priority of eventual unsatisfied person u,
                    # secondary key = priority of candidate itself.
                    cands_at_best_d.sort(
                        key=lambda item: (priority_rank[item[1]], priority_rank[item[0]])
                    )
                    newly_assigned[p] = cands_at_best_d[0][0]

            for p, target in newly_assigned.items():
                p_t[p] = target
                unassigned.remove(p)

        # Fallback for any person who cannot reach Ut directly or indirectly
        for p in list(unassigned):
            cands = C[p] - {p}
            if cands:
                p_t[p] = min(cands, key=lambda j: priority_rank[j])
            else:
                raise RuntimeError(
                    f"Algorithm invariant violated: individual '{p}' has no candidate pointees other than themselves."
                )
            unassigned.remove(p)

        # =====================================================================
        # 3. TRADING PHASE (Appendix A)
        # =====================================================================
        # Find all cycles in the directed graph p_t
        visited_global: Set[str] = set()
        cycles: List[List[str]] = []
        for p in remaining_people:
            if p not in visited_global:
                curr = p
                curr_visited: List[str] = []
                while curr not in visited_global and curr not in curr_visited:
                    curr_visited.append(curr)
                    curr = p_t[curr]
                if curr in curr_visited:
                    idx = curr_visited.index(curr)
                    cycle = curr_visited[idx:]
                    cycles.append(cycle)
                visited_global.update(curr_visited)

        # Execute trades prescribed by each cycle
        new_h = dict(h)
        for cycle in cycles:
            for i in range(len(cycle)):
                giver = p_t[cycle[i]]
                new_h[cycle[i]] = h[giver]

        prev_p = dict(p_t)
        prev_h = dict(h)
        h = new_h

    return assigned
