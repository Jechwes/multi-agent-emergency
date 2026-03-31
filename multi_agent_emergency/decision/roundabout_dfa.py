"""
roundabout_dfa.py
=================
Safety DFA for the roundabout driving task, compatible with
FMTensJelmar's ``DFATree`` from ``dfa_tree_r1_risk_min.py``.

Temporal-logic specification
----------------------------
    ψ = G( ¬collision ∧ ¬non_drivable ∧ ¬pedestrian )

The DFA has **two states**:

    q_safe (0) : accepting AND initial.  Normal driving.
    q_fail (1) : absorbing failure state (sink).  Entered on collision,
                 non-drivable violation, or pedestrian occupying the
                 same cell.

The safety filter prevents failures by accumulating risk over the
look-ahead horizon.  Each hazard (pedestrian, non-drivable, other
vehicle) carries a cost that grows with proximity.  When accumulated
risk exceeds the threshold, the filter overrides the DP action to
yield before the car reaches the hazard.

Four letters encode the atomic propositions:

    col 0 → 'drivable'      : all clear
    col 1 → 'non_drivable'  : vehicle on non-drivable lanelet  → fail
    col 2 → 'pedestrian'    : pedestrian occupies same cell     → fail
    col 3 → 'collision'     : two vehicles share a lanelet      → fail

Transition table ``trans[q, letter] → q'``:

    ┌──────────┬──────────┬──────────────┬─────────────┬────────────┐
    │          │ driv (0) │ non_driv (1) │ ped (2)     │ coll (3)   │
    ├──────────┼──────────┼──────────────┼─────────────┼────────────┤
    │ q_safe(0)│    0     │      1       │      1      │     1      │
    │ q_fail(1)│    1     │      1       │      1      │     1      │
    └──────────┴──────────┴──────────────┴─────────────┴────────────┘

DFA.F    = 0   (accepting = initial = safe, root of the DFA tree)
DFA.sink = 1   (absorbing failure)

Labelling function
------------------
``classify_joint_state()`` takes the lanelet indices of all vehicles
plus the pedestrian lanelet, and returns the worst-case letter over
the joint state.  In single-agent mode a single-element list is
passed and the collision check is trivially skipped.

Risk-cost association
---------------------
    'drivable'      → 0.0       (no penalty)
    'non_drivable'  → tuneable  (moderate)
    'pedestrian'    → tuneable  (high)
    'collision'     → tuneable  (high)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional

from abstraction.roundabout_abstraction import LaneletGraph


class RoundaboutDFA:
    """
    Two-state safety DFA:
        q_safe(0) — normal driving (accepting + initial)
        q_fail(1) — absorbing failure (sink)

    Any hazard (non-drivable, pedestrian, collision) transitions to
    q_fail.  The online safety filter prevents this by accumulating
    risk over the look-ahead horizon and making the car yield before
    it reaches any hazard.

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
        cost_collision: float = 300.0,
    ) -> None:
        # --- DFA states ---
        # q_safe = 0 : accepting + initial (normal driving)
        # q_fail = 1 : absorbing failure
        self.S  = [0, 1]
        self.F  = 0              # accepting state  (root of the DFA tree)
        self.sink = 1            # absorbing failure state
        self.S0 = 0              # initial state

        # --- Alphabet (4 letters) ---
        #   col 0 = 'drivable'      all safe
        #   col 1 = 'non_drivable'  vehicle on non-drivable lanelet
        #   col 2 = 'pedestrian'    pedestrian occupies same cell
        #   col 3 = 'collision'     two vehicles share a lanelet
        self.act = ['drivable', 'non_drivable', 'pedestrian', 'collision']

        # --- Transition table  (|S| × |letters|) ---
        self.trans = np.array([
            # drivable  non_drivable  pedestrian  collision
            [  0,          1,            1,          1  ],  # q_safe: any hazard → fail
            [  1,          1,            1,          1  ],  # q_fail: absorbing
        ], dtype=int)

        # --- Label → column mapping ---
        self._label_to_col: Dict[str, int] = {
            'drivable':      0,
            'non_drivable':  1,
            'pedestrian':    2,
            'collision':     3,
        }

        # --- Per-letter risk costs (used by the online safety filter) ---
        self.risk_cost: Dict[str, float] = {
            'drivable':      0.0,
            'non_drivable':  cost_non_drivable,
            'pedestrian':    cost_pedestrian,
            'collision':     cost_collision,
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
        """Map a label string to its transition-table column index."""
        if label not in self._label_to_col:
            raise KeyError(f"Unknown DFA label '{label}'. "
                           f"Valid labels: {list(self._label_to_col)}")
        return self._label_to_col[label]

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

    # ------------------------------------------------------------------
    # Joint-state labelling
    # ------------------------------------------------------------------

    def classify_joint_state(
        self,
        vehicle_lanelet_indices: List[int],
        graph: LaneletGraph,
        ped_lanelet_idx: Optional[int] = None,
    ) -> str:
        """
        Classify the joint state of all agents into a DFA letter.

        Works for both single-agent (list of one index) and multi-agent
        (list of N indices) scenarios.

        Check order reflects severity (highest first):
          1. collision    — any two vehicles share the same lanelet
          2. pedestrian   — a vehicle occupies the same lanelet as the
                            pedestrian
          3. non_drivable — any vehicle is on a non-drivable lanelet
          4. drivable     — all clear

        Parameters
        ----------
        vehicle_lanelet_indices
            Flat lanelet index for each vehicle.  For single-agent pass
            a one-element list; the collision check is then trivially
            skipped.
        graph
            LaneletGraph that knows which lanelets are drivable.
        ped_lanelet_idx
            Flat lanelet index of the pedestrian, or ``None`` if no
            pedestrian is present.

        Returns
        -------
        One of 'drivable', 'non_drivable', 'pedestrian', 'collision'.
        """
        # 1. Collision: two or more vehicles on the same lanelet
        if len(vehicle_lanelet_indices) > 1:
            if len(set(vehicle_lanelet_indices)) < len(vehicle_lanelet_indices):
                return 'collision'

        # 2. Pedestrian: a vehicle shares the same lanelet as the pedestrian
        if ped_lanelet_idx is not None:
            if ped_lanelet_idx in vehicle_lanelet_indices:
                return 'pedestrian'

        # 3. Any vehicle on a non-drivable lanelet
        for idx in vehicle_lanelet_indices:
            if not graph.is_lanelet_drivable(idx):
                return 'non_drivable'

        # 4. All clear
        return 'drivable'

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "RoundaboutDFA  (2-state)  –  q_safe / q_fail",
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
        return ("RoundaboutDFA(2-state: q_safe/q_fail, letters=4)")
