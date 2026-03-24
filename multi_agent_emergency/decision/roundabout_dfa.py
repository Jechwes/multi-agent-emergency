"""
roundabout_dfa.py
=================
Co-safety DFA for the roundabout driving task, compatible with
FMTensJelmar's ``DFATree`` from ``dfa_tree_r1_risk_min.py``.

Temporal-logic specification
----------------------------
The co-safety (guarantee) specification encodes:

    φ = G( ¬non_drivable  ∧  ¬pedestrian  ∧  ¬other_car )

i.e. **always** avoid non-drivable areas, pedestrians, and other cars.

The DFA has **two states**:

    q_safe (0) : accepting AND initial.  The car is operating safely.
                 Loops to itself on the 'safe' label.
    q_fail (1) : absorbing failure state (sink).  Entered when any
                 atomic proposition is violated.

Four letters encode the three atomic propositions plus the safe case:

    col 0 → 'safe'          : ¬nd ∧ ¬ped ∧ ¬car       → q_safe → q_safe
    col 1 → 'non_drivable'  : in a non-drivable area   → q_safe → q_fail
    col 2 → 'pedestrian'    : collision with pedestrian → q_safe → q_fail
    col 3 → 'other_car'     : collision with other car  → q_safe → q_fail

Transition table ``trans[q, letter] → q'``:

    ┌──────────┬──────────┬──────────────┬─────────────┬────────────┐
    │          │ safe (0) │ non_driv (1) │ ped (2)     │ car (3)    │
    ├──────────┼──────────┼──────────────┼─────────────┼────────────┤
    │ q_safe(0)│    0     │      1       │      1      │     1      │
    │ q_fail(1)│    1     │      1       │      1      │     1      │
    └──────────┴──────────┴──────────────┴─────────────┴────────────┘

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

Offline vs online labeling
--------------------------
The DFATree is solved **offline** with static label masks (road
geometry: which cells are drivable).  Since only letter 0 ('safe')
loops back to q_safe, the tree only ever uses the 'safe' mask — the
offline computation does not need to know where pedestrians or cars are.

The 4-letter alphabet is used **online** by the safety filter and the
DFA state tracker:
  - ``classify_state()`` checks all three atomic propositions and
    returns the appropriate letter.
  - The safety filter uses the associated ``risk_cost`` to grade the
    severity of different violations.
  - The DFA state tracker advances ``q_current`` based on the label.

This means the 4-letter alphabet adds **zero overhead** to the offline
tree-solve while giving the online safety filter a direct connection
to the temporal-logic specification.

Risk-cost association
---------------------
The ``risk_cost`` attribute maps each letter to a scalar cost:

    'safe'          → 0.0        (no penalty)
    'non_drivable'  → tuneable   (moderate penalty)
    'pedestrian'    → tuneable   (high penalty)
    'other_car'     → tuneable   (high penalty)

These are used by the online safety filter to evaluate the instantaneous
risk when the car's label transitions away from 'safe'.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional


class RoundaboutDFA:
    """
    Co-safety DFA for φ = G(¬non_drivable ∧ ¬pedestrian ∧ ¬other_car).

    Compatible with DFATree from dfa_tree_r1_risk_min.py.

    Attributes expected by DFATree
    ------------------------------
    S     : list of states [0, 1]
    F     : int, accepting state id  (= 0, also initial)
    sink  : int, absorbing failure   (= 1)
    trans : np.ndarray  (2, 4) with target state ids
    act   : list of human-readable letter names  (length 4)

    Safety-filter extras
    --------------------
    risk_cost : dict  letter_name → float  (severity cost for the filter)
    """

    def __init__(
        self,
        cost_non_drivable: float = 40.0,
        cost_pedestrian: float = 500.0,
        cost_other_car: float = 300.0,
    ) -> None:
        # --- DFA states ---
        # q_safe = 0 : accepting + initial (safe operation)
        # q_fail = 1 : absorbing failure
        self.S  = [0, 1]
        self.F  = 0              # accepting state  (root of the DFA tree)
        self.sink = 1            # absorbing failure state
        self.S0 = 0              # initial state

        # --- Alphabet (4 letters) ---
        #   col 0 = 'safe'          ¬nd ∧ ¬ped ∧ ¬car
        #   col 1 = 'non_drivable'  in a non-drivable area
        #   col 2 = 'pedestrian'    collision with pedestrian
        #   col 3 = 'other_car'     collision with other car
        self.act = ['safe', 'non_drivable', 'pedestrian', 'other_car']

        # --- Transition table  (|S| × |letters|) ---
        self.trans = np.array([
            # safe  non_drivable  pedestrian  other_car
            [  0,       1,            1,          1  ],  # q_safe
            [  1,       1,            1,          1  ],  # q_fail: absorbing
        ], dtype=int)

        # --- Label → column mapping ---
        self._label_to_col: Dict[str, int] = {
            'safe':          0,
            'non_drivable':  1,
            'pedestrian':    2,
            'other_car':     3,
        }

        # --- Per-letter risk costs (used by the online safety filter) ---
        self.risk_cost: Dict[str, float] = {
            'safe':          0.0,
            'non_drivable':  cost_non_drivable,
            'pedestrian':    cost_pedestrian,
            'other_car':     cost_other_car,
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

    def classify_state(
        self,
        d_ego: float,
        lane_half_width: float,
        lane_drivable: bool = True,
        ped_nearby: bool = False,
        car_nearby: bool = False,
    ) -> str:
        """
        Classify the current state into a DFA letter based on the
        three atomic propositions of the co-safety specification.

        The check order reflects severity (highest first):
          1. pedestrian collision  → 'pedestrian'
          2. other-car collision   → 'other_car'
          3. non-drivable area     → 'non_drivable'
          4. otherwise             → 'safe'

        Parameters
        ----------
        d_ego          : lateral deviation from lane centre [m]
        lane_half_width: half the lane width [m]
        lane_drivable  : whether the current lane is drivable
        ped_nearby     : True if a pedestrian is dangerously close
        car_nearby     : True if another car is dangerously close

        Returns
        -------
        One of 'safe', 'non_drivable', 'pedestrian', 'other_car'.
        """
        if ped_nearby:
            return 'pedestrian'
        if car_nearby:
            return 'other_car'
        if not lane_drivable or abs(d_ego) > lane_half_width:
            return 'non_drivable'
        return 'safe'

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "RoundaboutDFA  (co-safety)  –  φ = G(¬nd ∧ ¬ped ∧ ¬car)",
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
        return ("RoundaboutDFA(co-safety, "
                "G(¬nd ∧ ¬ped ∧ ¬car), states=2, letters=4)")
