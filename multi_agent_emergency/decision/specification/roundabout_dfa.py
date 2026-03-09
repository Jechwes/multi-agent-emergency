"""
roundabout_dfa.py
=================
Safety-filter DFA for the roundabout driving task, compatible with
FMTensJelmar's ``DFATree`` from ``dfa_tree_r1_risk_min.py``.

Safety-filter approach
----------------------
Instead of encoding reachability (F(t)), we use a pure **safety** spec:

    φ = G(safe)          – always remain in safe (drivable) regions

The DFA has **two states**:

    q_safe (0) : accepting AND initial.  The car is operating safely.
                 Loops to itself on the 'safe' label.
    q_fail (1) : absorbing failure state (sink).  Entered when the car
                 observes a risky / non-drivable label.

Multiple risk-level labels transition q_safe → q_fail, each carrying a
different associated cost (tuned to severity):

    col 0 → 'safe'      : drivable road, no risk        → q_safe → q_safe
    col 1 → 'risk_low'  : near lane edge, moderate risk → q_safe → q_fail
    col 2 → 'risk_high' : non-drivable area, high risk  → q_safe → q_fail

Transition table ``trans[q, letter] → q'``:

    ┌──────────┬──────────┬────────────┬─────────────┐
    │          │ safe (0) │ risk_low(1)│ risk_high(2)│
    ├──────────┼──────────┼────────────┼─────────────┤
    │ q_safe(0)│    0     │     1      │      1      │
    │ q_fail(1)│    1     │     1      │      1      │
    └──────────┴──────────┴────────────┴─────────────┘

DFA.F    = 0   (accepting=initial=safe, root of the DFA tree)
DFA.sink = 1   (absorbing failure)

DFA-tree structure
------------------
Because only ``trans[0, 0] = 0`` maps back to q_safe, ``initiate()``
creates a single child of the root (q_safe, edge='safe').  Every
``grow()`` call extends the chain by one node:

    root(q0) ──safe──▶ n₁(q0) ──safe──▶ n₂(q0) ──safe──▶  …

The tree depth equals the safety look-ahead horizon.  At each level the
value update applies:

    V_n(s') = γ · L_safe(s') · [ V_parent(s') + c(s') ] · P^{π}

so cells outside the 'safe' mask are zeroed, while the cost map c(s')
creates a smooth risk-gradient inside the safe zone.

Risk-cost association
---------------------
The ``risk_cost`` attribute maps each letter to a scalar cost:

    'safe'      → 0.0        (no penalty)
    'risk_low'  → tuneable   (moderate penalty)
    'risk_high' → tuneable   (high penalty)

These are used by the online safety filter to evaluate the instantaneous
risk when the car's label transitions away from 'safe'.
"""

from __future__ import annotations
import numpy as np
from typing import Dict


class RoundaboutDFA:
    """
    Safety-filter DFA (0-based) for the DFATree from dfa_tree_r1_risk_min.py.

    Attributes expected by DFATree
    ------------------------------
    S     : list of states [0, 1]
    F     : int, accepting state id  (= 0, also initial)
    sink  : int, absorbing failure   (= 1)
    trans : np.ndarray  (2, 3) with target state ids
    act   : list of human-readable letter names  (length 3)

    Safety-filter extras
    --------------------
    risk_cost : dict  letter_name → float  (severity cost for the filter)
    """

    def __init__(
        self,
        cost_risk_low: float = 5.0,
        cost_risk_high: float = 50.0,
    ) -> None:
        # --- DFA states ---
        # q_safe = 0 : accepting + initial (safe operation)
        # q_fail = 1 : absorbing failure
        self.S  = [0, 1]
        self.F  = 0              # accepting state  (root of the DFA tree)
        self.sink = 1            # absorbing failure state
        self.S0 = 0              # initial state

        # --- Alphabet ---
        #   col 0 = 'safe'       drivable road, no risk
        #   col 1 = 'risk_low'   near lane edge
        #   col 2 = 'risk_high'  non-drivable / boundary
        self.act = ['safe', 'risk_low', 'risk_high']

        # --- Transition table  (|S| × |letters|) ---
        self.trans = np.array([
            # safe  risk_low  risk_high
            [  0,      1,         1  ],   # q_safe: stays safe only on 'safe'
            [  1,      1,         1  ],   # q_fail: absorbing
        ], dtype=int)

        # --- Label → column mapping ---
        self._label_to_col: Dict[str, int] = {
            'safe':      0,
            'risk_low':  1,
            'risk_high': 2,
        }

        # --- Per-letter risk costs (used by the online safety filter) ---
        self.risk_cost: Dict[str, float] = {
            'safe':      0.0,
            'risk_low':  cost_risk_low,
            'risk_high': cost_risk_high,
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
        """Map a cell label string to the transition-table column index."""
        return self._label_to_col.get(label, 0)   # default → 'safe'

    def next_state(self, q: int, label: str) -> int:
        """Deterministic DFA transition."""
        col = self.label_to_column(label)
        return int(self.trans[q, col])

    def is_accepting(self, q: int) -> bool:
        return q == self.F

    def is_trap(self, q: int) -> bool:
        return q == self.sink

    def get_risk_cost(self, label: str) -> float:
        """Return the risk cost associated with a label."""
        return self.risk_cost.get(label, 0.0)

    def classify_lateral(self, d: float, lane_half_width: float,
                         risk_low_fraction: float = 0.75) -> str:
        """
        Classify a lateral deviation *d* into a DFA letter.

        Parameters
        ----------
        d                 : lateral deviation from lane centre [m]
        lane_half_width   : half the lane width [m]
        risk_low_fraction : |d|/half_width above which → 'risk_low'
                            (below → 'safe', at the edge → 'risk_high')

        Returns
        -------
        One of 'safe', 'risk_low', 'risk_high'.
        """
        ratio = abs(d) / lane_half_width if lane_half_width > 0 else 0.0
        if ratio >= 1.0:
            return 'risk_high'
        elif ratio >= risk_low_fraction:
            return 'risk_low'
        else:
            return 'safe'

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "RoundaboutDFA  (safety filter)  –  φ = G(safe)",
            f"  States : {self.S}",
            f"  Initial: {self.S0}  (= accepting)",
            f"  Accept : {self.F}",
            f"  Sink   : {self.sink}",
            f"  Letters: {self.act}",
            f"  Risk costs: {self.risk_cost}",
            "  trans =",
        ]
        for q in self.S:
            lines.append(f"    q{q}: {list(self.trans[q])}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "RoundaboutDFA(safety_filter, G(safe), states=2, letters=3)"
