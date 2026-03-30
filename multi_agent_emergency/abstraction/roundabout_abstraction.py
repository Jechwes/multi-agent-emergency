"""
roundabout_abstraction.py
=========================
Build a **lanelet-graph** MDP abstraction for the roundabout.

Lanelet-graph approach
----------------------
Each lanelet is identified by a flat index

    idx = section * n_lanes + lane

The state space is the set of all lanelets (N_sections × N_lanes).
Each controlled vehicle has its own transition matrix of shape
``(N_lanelets, N_lanelets * N_actions)`` in the block-column layout
expected by ``SysAbs1D``.

Actions: advance, lane_in, lane_out, yield.

Pedestrian dimension
--------------------
A decoupled dimension models a crossing pedestrian as a Markov chain.
The pedestrian's radial position is discretised over the full
roundabout width.

Cost design
-----------
*Higher cost = higher risk.*

**State cost:** zero for drivable lanelets, high penalty for
non-drivable ones.

**Action cost:** negative on advance (reward forward progress),
small positive on lane changes, zero on yield.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Minimal SysAbs wrapper for DFATree compatibility
# ---------------------------------------------------------------------------

class SysAbs1D:
    """
    Thin adapter that exposes the interface expected by
    ``DFATree`` from ``dfa_tree_r1_risk_min.py``.

    Required attributes
    -------------------
    .P       : ndarray (N, N*nu)  transition matrix (block-column layout)
    .P_flat  : same reference as .P
    .dim     : int = 1
    """

    def __init__(self, P_flat: np.ndarray) -> None:
        self.P = P_flat
        self.P_flat = P_flat
        self.dim = 1


# ---------------------------------------------------------------------------
# Lanelet connectivity graph
# ---------------------------------------------------------------------------

class LaneletGraph:
    """
    Graph of lanelet connectivity for a roundabout.

    Each lanelet is identified by a flat index:

        idx = section * n_lanes + lane
        (section, lane) = (idx // n_lanes, idx % n_lanes)

    Parameters
    ----------
    n_sections     : number of angular sectors.
    n_lanes        : total number of concentric rings.
    drivable_lanes : which lane indices are normally drivable (e.g. [1, 2]).
    inner_radius   : inner edge of lane 0 [m].
    lane_width     : radial width of each lane [m].
    """

    def __init__(
        self,
        n_sections: int,
        n_lanes: int,
        drivable_lanes: List[int],
        inner_radius: float,
        lane_width: float,
    ) -> None:
        self.n_sections = n_sections
        self.n_lanes = n_lanes
        self.n_lanelets = n_sections * n_lanes
        self.drivable_lanes = set(drivable_lanes)
        self.inner_radius = inner_radius
        self.lane_width = lane_width

    # -- flat index helpers --------------------------------------------------

    def to_index(self, section: int, lane: int) -> int:
        """Convert (section, lane) to flat lanelet index."""
        return section * self.n_lanes + lane

    def from_index(self, idx: int) -> Tuple[int, int]:
        """Convert flat lanelet index to (section, lane)."""
        return (idx // self.n_lanes, idx % self.n_lanes)

    # -- queries -------------------------------------------------------------

    def is_drivable(self, lane_id: int) -> bool:
        """Check whether a *lane* index (not lanelet) is drivable."""
        return lane_id in self.drivable_lanes

    def is_lanelet_drivable(self, idx: int) -> bool:
        """Check whether a lanelet (flat index) is drivable."""
        _, lane = self.from_index(idx)
        return lane in self.drivable_lanes

    def adjacent_lanes(self, lane_id: int) -> List[int]:
        """Return all valid adjacent lane ids (drivable or not)."""
        result = []
        for dl in (-1, +1):
            adj = lane_id + dl
            if 0 <= adj < self.n_lanes:
                result.append(adj)
        return result

    def successor_section(self, section: int) -> int:
        """Next section (wraps around the roundabout)."""
        return (section + 1) % self.n_sections

    # -- transition matrix ---------------------------------------------------

    # Action indices (fixed ordering)
    ACTION_ADVANCE  = 0
    ACTION_LANE_IN  = 1   # toward lane 0 (inner)
    ACTION_LANE_OUT = 2   # toward lane n_lanes-1 (outer)
    ACTION_YIELD    = 3
    N_ACTIONS       = 4

    def build_transition_matrix(
        self,
        process_noise: float = 0.05,
    ) -> np.ndarray:
        """
        Build the lanelet-level transition matrix.

        For each lanelet and each of the 4 actions the method computes a
        probability distribution over successor lanelets.

        Actions
        -------
        0 - advance   : move to the same lane in the next section (wraps).
        1 - lane_in   : move one lane inward (toward lane 0).
                         If already at lane 0 → stays in place.
        2 - lane_out  : move one lane outward (toward lane n_lanes-1).
                         If already at outermost lane → stays in place.
        3 - yield      : stay in the current lanelet.

        Stochastic noise
        ----------------
        With probability ``process_noise`` the vehicle drifts to an
        adjacent lane (split equally inward / outward) instead of
        reaching the nominal successor.  At lane boundaries the
        impossible drift direction is folded back into the nominal
        successor.

        Returns
        -------
        P_flat : ndarray, shape (N_lanelets, N_lanelets * N_ACTIONS)
            Block-column layout: columns
            ``[a * N_lanelets : (a+1) * N_lanelets]`` hold the
            transition probabilities under action *a*.
        """
        N = self.n_lanelets
        P_flat = np.zeros((N, N * self.N_ACTIONS), dtype=float)

        for idx in range(N):
            section, lane = self.from_index(idx)
            next_sec = self.successor_section(section)

            for action in range(self.N_ACTIONS):
                col_offset = action * N

                # --- determine nominal successor lanelet ---
                if action == self.ACTION_ADVANCE:
                    nom_section = next_sec
                    nom_lane = lane
                elif action == self.ACTION_LANE_IN:
                    nom_section = section
                    nom_lane = lane - 1 if lane > 0 else lane
                elif action == self.ACTION_LANE_OUT:
                    nom_section = section
                    nom_lane = lane + 1 if lane < self.n_lanes - 1 else lane
                else:  # ACTION_YIELD
                    nom_section = section
                    nom_lane = lane

                nom_idx = self.to_index(nom_section, nom_lane)

                # --- distribute probability with lane drift noise ---
                noise = float(np.clip(process_noise, 0.0, 0.49))
                p_main = 1.0 - 2.0 * noise
                p_in = noise    # drift toward lane 0
                p_out = noise   # drift toward n_lanes-1

                # Drift targets share the same section as the nominal
                drift_in_lane = nom_lane - 1
                drift_out_lane = nom_lane + 1

                # Fold impossible drifts back into main
                if drift_in_lane < 0:
                    p_main += p_in
                    p_in = 0.0
                if drift_out_lane >= self.n_lanes:
                    p_main += p_out
                    p_out = 0.0

                P_flat[idx, col_offset + nom_idx] += p_main
                if p_in > 0.0:
                    P_flat[idx, col_offset + self.to_index(nom_section, drift_in_lane)] += p_in
                if p_out > 0.0:
                    P_flat[idx, col_offset + self.to_index(nom_section, drift_out_lane)] += p_out

        return P_flat


# ---------------------------------------------------------------------------
# Label matrix builder
# ---------------------------------------------------------------------------

def build_label_matrix(
    graph: LaneletGraph,
    n_letters: int = 4,
) -> np.ndarray:
    """
    Build a label indicator matrix for a single vehicle dimension.

    Shape: ``(n_letters, N_lanelets)``.  Each column encodes which
    atomic proposition holds at that lanelet.

    Alphabet
    --------
    0 - 'drivable'      lanelet whose lane ∈ drivable_lanes
    1 - 'non_drivable'  lanelet whose lane ∉ drivable_lanes
    2 - 'pedestrian'    (unused here — set by pedestrian dimension)
    3 - 'collision'     (unused here — emerges from joint label
                         when two vehicles share a lanelet)

    Collision detection is not encoded per-vehicle; it arises in the
    product automaton when the DFA observes the combined label across
    all vehicle dimensions.
    """
    N = graph.n_lanelets
    L = np.zeros((n_letters, N), dtype=float)

    for idx in range(N):
        if graph.is_lanelet_drivable(idx):
            L[0, idx] = 1.0   # drivable
        else:
            L[1, idx] = 1.0   # non_drivable

    return L


# ---------------------------------------------------------------------------
# Cost map builder
# ---------------------------------------------------------------------------

def build_state_cost(
    graph: LaneletGraph,
    non_drivable_penalty: float = 50.0,
) -> np.ndarray:
    """
    State cost vector of length ``N_lanelets`` for one vehicle.

    Zero for drivable lanelets, ``non_drivable_penalty`` for the rest.
    Different penalty values for ego vs opponent enable different risk
    thresholds.
    """
    N = graph.n_lanelets
    cost = np.zeros(N, dtype=float)
    for idx in range(N):
        if not graph.is_lanelet_drivable(idx):
            cost[idx] = non_drivable_penalty
    return cost


# ---------------------------------------------------------------------------
# Action cost builder
# ---------------------------------------------------------------------------

def build_action_cost(
    advance_reward: float = -1.0,
    lane_change_cost: float = 0.5,
) -> np.ndarray:
    """
    Action cost vector of length ``N_ACTIONS`` (4) for one controlled vehicle.

    Index mapping follows ``LaneletGraph`` action constants:

    0 - advance   : negative cost (reward for forward progress)
    1 - lane_in   : small positive cost (penalise risky manoeuvre)
    2 - lane_out  : small positive cost
    3 - yield     : zero cost
    """
    cost = np.zeros(LaneletGraph.N_ACTIONS, dtype=float)
    cost[LaneletGraph.ACTION_ADVANCE]  = advance_reward
    cost[LaneletGraph.ACTION_LANE_IN]  = lane_change_cost
    cost[LaneletGraph.ACTION_LANE_OUT] = lane_change_cost
    cost[LaneletGraph.ACTION_YIELD]    = 0.0
    return cost


# ---------------------------------------------------------------------------
# Pedestrian dimension
# ---------------------------------------------------------------------------

def _build_pedestrian_labels(
    centres_p: np.ndarray,
    lane_width: float,
    n_lanes: int,
    ref_lane: int,
    n_letters: int = 4,
) -> np.ndarray:
    """Label matrix for pedestrian dimension.

    When the pedestrian is on the reference (driving) lane, the label
    is 'pedestrian' (col 2).  Otherwise 'safe' (col 0).
    """
    d_low = (ref_lane - n_lanes / 2.0) * lane_width
    d_high = d_low + lane_width

    L_p = np.zeros((n_letters, len(centres_p)), dtype=float)
    for i, dp in enumerate(centres_p):
        if d_low <= dp < d_high:
            L_p[2, i] = 1.0   # pedestrian on driving lane
        else:
            L_p[0, i] = 1.0   # safe
    return L_p


def _build_pedestrian_cost(
    centres_p: np.ndarray,
    lane_width: float,
    n_lanes: int,
    ref_lane: int,
    penalty: float = 50.0,
) -> np.ndarray:
    """State cost for pedestrian dimension (on reference lane)."""
    d_low = (ref_lane - n_lanes / 2.0) * lane_width
    d_high = d_low + lane_width
    cost_p = np.zeros(len(centres_p), dtype=float)
    for i, dp in enumerate(centres_p):
        if d_low <= dp < d_high:
            cost_p[i] = penalty
    return cost_p


def build_pedestrian_chain(
    N_p: int,
    lane_width: float,
    n_lanes: int = 4,
    p_move: float = 0.3,
) -> Dict:
    """
    Markov chain for pedestrian crossing inward from outer edge.

    Returns dict with P_p, centres_p, rho_p.
    """
    lateral_half = (n_lanes / 2.0) * lane_width
    cell_w = 2.0 * lateral_half / N_p
    centres_p = np.linspace(-lateral_half + cell_w / 2.0,
                             lateral_half - cell_w / 2.0, N_p)

    P_p = np.zeros((N_p, N_p), dtype=float)
    for i in range(N_p):
        if i > 0:
            P_p[i, i - 1] = p_move
            P_p[i, i]     = 1.0 - p_move
        else:
            P_p[0, 0] = 1.0

    rho_p = np.zeros(N_p, dtype=float)
    for i, dp in enumerate(centres_p):
        if dp >= lane_width:
            rho_p[i] = 1.0
    if rho_p.sum() > 0:
        rho_p /= rho_p.sum()
    else:
        rho_p[:] = 1.0 / N_p

    return {'P_p': P_p, 'centres_p': centres_p, 'rho_p': rho_p}
