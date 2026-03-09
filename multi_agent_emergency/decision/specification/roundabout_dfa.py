"""
roundabout_dfa.py
=================
DFA specification for the roundabout driving task, compatible with 
FMTensJelmar's ``DFATree`` from ``dfa_tree_r1_risk_min.py``.

Specification
-------------
The DFA encodes the **conjunction** of two objectives:

  1. **Safety**:  G(¬n ∧ ¬c)
     Always avoid non-drivable regions (n) and collisions (c).

  2. **Reachability**:  F(t)
     Eventually reach the target/exit section (t).

Combined (product) specification:
    φ = G(¬n ∧ ¬c) ∧ F(t)

Atomic propositions (alphabet)
------------------------------
Each cell of the discretised roundabout is labelled with a *set* of APs.
We use the following:

    'r'  – drivable roundabout road (safe, not target)
    'n'  – non-drivable area (violation)
    'c'  – collision zone (violation – reserved for dynamic obstacles)
    't'  – target/exit region

The DFA below is the **product** of the safety and reachability DFAs:

    State 0 (q_safe):  "safe, target not yet reached"      – initial state
    State 1 (q_acc) :  "safe, target reached"               – accepting (F)
    State 2 (q_sink):  "violation occurred"                  – absorbing trap

Alphabet columns (letters) – the DFATree expects ``DFA.trans`` with shape
``(|S|, |act|)`` where columns are logical predicates over AP observations.

We define four alphabet letters (columns of ``trans``):

    col 0 → 'r' : road, no violation, not target
    col 1 → 't' : target reached, no violation
    col 2 → 'n' : non-drivable violation
    col 3 → 'c' : collision violation

Transition table ``trans[q, letter] → q'``:

    ┌──────────┬───────┬───────┬───────┬───────┐
    │          │ r (0) │ t (1) │ n (2) │ c (3) │
    ├──────────┼───────┼───────┼───────┼───────┤
    │ q0 (0)   │   0   │   1   │   2   │   2   │
    │ q1 (1)   │   1   │   1   │   2   │   2   │
    │ q2 (2)   │   2   │   2   │   2   │   2   │
    └──────────┘───────┘───────┘───────┘───────┘

DFA.F = 1  (accepting state, the root of the DFA tree)
DFA.sink = 2  (absorbing trap)
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional


class RoundaboutDFA:
    """
    A 0-based DFA compatible with ``DFATree`` from dfa_tree_r1_risk_min.py.

    Attributes expected by DFATree
    ------------------------------
    S     : list of states [0, 1, ..., nq-1]
    F     : int, accepting state id (0-based)
    sink  : int, absorbing trap state id (0-based)
    trans : np.ndarray  (|S|, |letters|) with target state ids
    act   : list of human-readable letter names (length == # columns)

    Additionally we store the label→column mapping so the labelling
    utilities can convert region labels to column indices.
    """

    def __init__(self) -> None:
        # States: 0 = safe/initial, 1 = accepting (target reached), 2 = trap
        self.S = [0, 1, 2]
        self.F = 1              # accepting state (root of DFA tree)
        self.sink = 2           # absorbing trap
        self.S0 = 0             # initial state

        # Alphabet: human-readable names per column
        self.act = ['r', 't', 'n', 'c']

        # Transition table  (|S| × |letters|)
        # trans[q, letter_col] = next_q
        self.trans = np.array([
            # r  t  n  c
            [0, 1, 2, 2],   # q0: safe, target not reached
            [1, 1, 2, 2],   # q1: safe, target reached (accepting)
            [2, 2, 2, 2],   # q2: trap (violation)
        ], dtype=int)

        # Convenient reverse map: label_string → column index
        self._label_to_col: Dict[str, int] = {
            'r': 0,
            't': 1,
            'n': 2,
            'c': 3,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def n_states(self) -> int:
        return len(self.S)

    @property
    def n_letters(self) -> int:
        return self.trans.shape[1]

    def label_to_column(self, label: str) -> int:
        """Map a cell label string ('r', 't', 'n', 'c') to the column index."""
        return self._label_to_col.get(label, 0)  # default to 'r' (safe road)

    def next_state(self, q: int, label: str) -> int:
        """Deterministic DFA transition."""
        col = self.label_to_column(label)
        return int(self.trans[q, col])

    def is_accepting(self, q: int) -> bool:
        return q == self.F

    def is_trap(self, q: int) -> bool:
        return q == self.sink

    def summary(self) -> str:
        lines = [
            "RoundaboutDFA  –  φ = G(¬n ∧ ¬c) ∧ F(t)",
            f"  States : {self.S}",
            f"  Initial: {self.S0}",
            f"  Accept : {self.F}",
            f"  Trap   : {self.sink}",
            f"  Letters: {self.act}",
            "  trans =",
        ]
        for q in self.S:
            lines.append(f"    q{q}: {list(self.trans[q])}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "RoundaboutDFA(G(¬n∧¬c)∧F(t), states=3, letters=4)"
