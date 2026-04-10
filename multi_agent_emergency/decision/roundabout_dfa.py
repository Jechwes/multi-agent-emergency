"""
roundabout_dfa.py
=================
Safety DFA for the roundabout driving task, compatible with
FMTensJelmar's ``DFATree`` from ``dfa_tree_r1_risk_min.py``.

Temporal-logic specification
----------------------------
The full safety specification is:

    ψ = G( ¬nd ∧ ¬ped ∧ ¬coll )

This is split between offline and online enforcement:

  **Offline (DFA tree):** G(¬nd) — avoid non-drivable areas.
    This is a per-dimension predicate (depends only on the ego
    vehicle's lanelet) and is handled by the rank-1 value iteration.

  **Online (safety filter):** G(¬ped ∧ ¬coll) — avoid pedestrians
    and inter-vehicle collisions.  Both are cross-dimensional
    predicates (depend on joint states of multiple agents) that
    cannot be rank-1 decomposed.

DFA structure (co-safety formulation)
-------------------------------------
Two states, two letters:

    q_safe (0) : initial, non-accepting.  Normal driving.
    q_fail (1) : accepting AND absorbing (sink).  Entered when
                 the vehicle is on a non-drivable lanelet.

    col 0 → 'safe'  : vehicle on a drivable lanelet
    col 1 → 'nd'    : vehicle on a non-drivable lanelet  → fail

Transition table ``trans[q, letter] → q'``:

    ┌──────────┬──────────┬──────────┐
    │          │ safe (0) │  nd (1)  │
    ├──────────┼──────────┼──────────┤
    │ q_safe(0)│    0     │    1     │
    │ q_fail(1)│    1     │    1     │
    └──────────┴──────────┴──────────┘

Online labelling
----------------
``classify_joint_state()`` is used by the online control loop for
DFA state tracking.  It returns the worst-case label including
cross-dimensional checks (pedestrian, collision) that are not part
of the DFA alphabet but are needed for the safety filter.

Risk-cost association (used by the online safety filter)
--------------------------------------------------------
    'safe'       → 0.0       (no penalty)
    'nd'         → tuneable  (moderate)
    'pedestrian' → tuneable  (high — online only)
    'collision'  → tuneable  (high — online only)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional

from abstraction.roundabout_abstraction import LaneletGraph


class RoundaboutDFA:
    """
    Two-state safety DFA with 2-letter alphabet for offline VI:
        q_safe(0) — normal driving (initial)
        q_fail(1) — absorbing failure (accepting, root of DFA tree)

    The offline DFA only encodes the per-dimension predicate G(¬nd).
    Pedestrian and collision avoidance are handled entirely by the
    online safety filter.

    Attributes expected by DFATree
    ------------------------------
    S     : list of states [0, 1]
    F     : int, accepting state id  (= 1)
    sink  : int, absorbing failure   (= 1)
    trans : np.ndarray  (2, 2) with target state ids
    act   : list of human-readable letter names  (length 2)

    Safety-filter extras
    --------------------
    risk_cost : dict  label → float  (includes online-only labels)
    """

    def __init__(
        self,
        cost_non_drivable: float = 40.0,
        cost_pedestrian: float = 500.0,
        cost_collision: float = 300.0,
    ) -> None:
        # --- DFA states ---
        self.S  = [0, 1]
        self.F  = 1              # accepting state  (root of the DFA tree)
        self.sink = 1            # absorbing failure state
        self.S0 = 0              # initial state

        # --- Alphabet (2 letters for offline DFA) ---
        self.act = ['safe', 'nd']

        # --- Transition table  (|S| × |letters|) ---
        self.trans = np.array([
            # safe  nd
            [  0,    1  ],   # q_safe: nd → fail
            [  1,    1  ],   # q_fail: absorbing
        ], dtype=int)

        # --- Label → column mapping ---
        self._label_to_col: Dict[str, int] = {
            'safe':  0,
            'nd':    1,
        }

        # --- Per-label risk costs (used by the online safety filter) ---
        # 'pedestrian' and 'collision' are included for the safety
        # filter even though they are not part of the DFA alphabet.
        self.risk_cost: Dict[str, float] = {
            'safe':       0.0,
            'nd':         cost_non_drivable,
            'pedestrian': cost_pedestrian,
            'collision':  cost_collision,
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
        """Map a DFA label string to its transition-table column index.

        Only 'safe' and 'nd' are valid DFA letters.  Online-only labels
        ('pedestrian', 'collision') raise KeyError.
        """
        if label not in self._label_to_col:
            raise KeyError(f"Unknown DFA label '{label}'. "
                           f"Valid DFA labels: {list(self._label_to_col)}")
        return self._label_to_col[label]

    def next_state(self, q: int, label: str) -> int:
        """Deterministic DFA transition.

        Online-only labels ('pedestrian', 'collision') always transition
        to q_fail since they represent safety violations.
        """
        if label in ('pedestrian', 'collision'):
            return self.sink
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
        Classify the joint state into a label for online DFA tracking.

        Returns the worst-case label.  'collision' and 'pedestrian' are
        online-only labels (not DFA letters) but ``next_state()`` handles
        them correctly by mapping to q_fail.

        Check order reflects severity (highest first):
          1. collision    — two vehicles share the same lanelet
          2. pedestrian   — a vehicle shares the pedestrian's lanelet
          3. nd           — any vehicle on a non-drivable lanelet
          4. safe         — all clear

        Returns
        -------
        One of 'collision', 'pedestrian', 'nd', 'safe'.
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
                return 'nd'

        # 4. All clear
        return 'safe'

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "RoundaboutDFA  (2-state, 2-letter)  –  q_safe / q_fail",
            f"  States : {self.S}",
            f"  Initial: {self.S0}",
            f"  Accept : {self.F}",
            f"  Sink   : {self.sink}",
            f"  DFA letters: {self.act}",
            f"  Risk costs (incl. online-only): {self.risk_cost}",
            "  trans =",
        ]
        for q in self.S:
            lines.append(f"    q{q}: {list(self.trans[q])}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "RoundaboutDFA(2-state, 2-letter: safe/nd)"
