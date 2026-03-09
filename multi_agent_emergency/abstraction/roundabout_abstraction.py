"""
roundabout_abstraction.py
=========================
Build decoupled 1-D MDP transition matrices and **risk-based cost maps**
for the roundabout, compatible with ``SimpleDPSolver`` from
``decision/simple_dp.py``.

Grid layout
-----------
The roundabout is divided into **sections** (angular sectors) and **lanes**
(concentric rings).  The car's continuous state is expressed in Frenet
coordinates with a *global* lateral reference:

    s ∈ [0, L_section]    longitudinal progress within the current section
    d ∈ [-N/2·w, +N/2·w]  lateral deviation from the midpoint between
                           drivable lanes 1 and 2  (positive = outward)

For a 4-lane roundabout with lane_width *w*:

    Lane 0  d ∈ [-2w, -w]   island / inner boundary   → risk_high
    Lane 1  d ∈ [-w ,  0]   inner drivable lane       → safe
    Lane 2  d ∈ [ 0 , +w]   outer drivable lane       → safe
    Lane 3  d ∈ [+w , +2w]  outer boundary            → risk_high

We discretise each axis independently into ``N_s`` and ``N_d`` grid cells
and build *deterministic* single-integrator transition matrices:

    P_s:  (N_s, N_s * nu_s)      longitudinal
    P_d:  (N_d, N_d * nu_d)      lateral

Pedestrian dimension
--------------------
A third decoupled dimension models a crossing pedestrian as a Markov
chain.  The pedestrian's lateral position is discretised over the same
4-lane span.  It starts on lane 3 and moves toward lane 0 with some
probability at each step.  When the pedestrian is on the drivable lanes
(1 or 2), the label is ``risk_high``; otherwise ``safe``.

Risk-based cost design
----------------------
*Higher cost ≡ higher risk.*

**Longitudinal (s-axis)**

    • state cost = 0  for all cells (the s-axis wraps around the section;
      all cells are drivable road).
    • action cost = ``−k_speed · v_s``  for each discrete speed level.
      Choosing higher forward speed yields a more negative (lower) cost,
      so value iteration naturally selects "go forward fast".

**Lateral (d-axis)**

    • state cost = negative (reward) inside the drivable corridor
      (lanes 1 and 2), rising to a large positive penalty in the
      non-drivable lanes (0 and 3).
    • action cost = small ``k_lat · |v_d|`` to discourage lateral wobble.

The value-iteration objective *minimises* the infinite-horizon discounted
sum of these costs → the car drives forward while staying centred.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

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
    .P       : ndarray (N, N*nu)   – transition matrix (block-column layout)
    .P_flat  : same reference as .P
    .dim     : int = 1
    """

    def __init__(self, P_flat: np.ndarray) -> None:
        self.P = P_flat
        self.P_flat = P_flat
        self.dim = 1


# ---------------------------------------------------------------------------
# 1-D deterministic transition matrix builder
# ---------------------------------------------------------------------------

def _build_1d_transitions(
    N: int,
    x_min: float,
    x_max: float,
    dt: float,
    speed_values: np.ndarray,
    wrap: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a deterministic transition matrix for a 1-D single-integrator:

        x(k+1) = x(k) + v · dt

    Parameters
    ----------
    N            : number of grid cells.
    x_min, x_max : physical axis extent.
    dt           : time-step [s].
    speed_values : (nu,) array of discrete speed actions.
    wrap         : True → periodic boundary (for s-axis).

    Returns
    -------
    P_flat  : (N, N*nu) deterministic transition matrix.
    centres : (N,) cell centres.
    """
    nu = len(speed_values)
    cell_w = (x_max - x_min) / N
    centres = np.linspace(x_min + cell_w / 2, x_max - cell_w / 2, N)

    P_flat = np.zeros((N, N * nu), dtype=float)

    for a_idx, v in enumerate(speed_values):
        for i in range(N):
            x_next = centres[i] + v * dt

            if wrap:
                x_next = x_min + (x_next - x_min) % (x_max - x_min)

            j = int(np.clip(round((x_next - centres[0]) / cell_w), 0, N - 1))
            P_flat[i, a_idx * N + j] = 1.0

    return P_flat, centres


# ---------------------------------------------------------------------------
# Cost-map builder
# ---------------------------------------------------------------------------

def _build_cost_maps(
    centres_s: np.ndarray,
    centres_d: np.ndarray,
    acc_s: np.ndarray,
    acc_d: np.ndarray,
    lane_width: float,
    n_lanes: int = 4,
    k_speed: float = 0.5,
    k_lat: float = 0.1,
    road_reward: float = -1.0,
    edge_penalty: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the risk-based cost arrays.

    The lateral cost profile is flat (``road_reward``) inside the drivable
    corridor (lanes 1 and 2, i.e. ``|d| ≤ lane_width``) and ramps up
    quadratically to ``edge_penalty`` in the non-drivable outer lanes
    (0 and 3).

    Returns
    -------
    state_cost_s  : (N_s,)   – zero everywhere (all road).
    action_cost_s : (nu_s,)  – negative proportional to forward speed.
    state_cost_d  : (N_d,)   – reward inside drivable corridor, penalty outside.
    action_cost_d : (nu_d,)  – small penalty for lateral speed magnitude.
    """
    N_s = len(centres_s)

    # --- Longitudinal costs ---
    state_cost_s = np.zeros(N_s, dtype=float)
    action_cost_s = -k_speed * acc_s.copy()

    # --- Lateral costs ---
    # Drivable corridor: |d| <= lane_width  (lanes 1 and 2 combined).
    # Outside that (lanes 0 and 3): rising penalty.
    drivable_half = lane_width                               # half-width of drivable region
    dist_outside = np.maximum(0.0, np.abs(centres_d) - drivable_half)
    max_outside  = max((n_lanes / 2.0 - 1.0) * lane_width, 1e-6)
    normalised   = dist_outside / max_outside                # 0 inside → 1 at span edge

    state_cost_d = road_reward + (edge_penalty - road_reward) * normalised ** 2

    # Small penalty for lateral speed magnitude (discourages wobble)
    action_cost_d = k_lat * np.abs(acc_d)

    return state_cost_s, action_cost_s, state_cost_d, action_cost_d


# ---------------------------------------------------------------------------
# Label-matrix builder  (for DFATree)
# ---------------------------------------------------------------------------

def build_label_matrices(
    N_s: int,
    N_d: int,
    centres_d: np.ndarray,
    lane_width: float,
    n_letters: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-dimension label indicator matrices for the DFATree.

    Alphabet columns (matching the safety-filter ``RoundaboutDFA``):

        0 → 'safe'       drivable road (lanes 1 and 2)
        1 → 'risk_low'   (reserved, not used by lane-level labelling)
        2 → 'risk_high'  non-drivable (lanes 0 and 3)

    The labelling is based on **lane boundaries**: the drivable corridor
    is ``|d| < lane_width`` (lanes 1 and 2 combined, centred at d = 0).
    Cells outside that corridor fall in the risky outer lanes.

    Parameters
    ----------
    N_s        : number of longitudinal cells.
    N_d        : number of lateral cells.
    centres_d  : (N_d,) array of lateral cell-centre positions.
    lane_width : single-lane radial width [m].
    n_letters  : number of alphabet symbols (default 3).

    Returns
    -------
    L_s : (n_letters, N_s)  – all 'safe' along s-axis.
    L_d : (n_letters, N_d)  – 'safe' in the drivable corridor,
                               'risk_high' in the boundary lanes.
    """
    # --- Longitudinal: entire section is drivable → 'safe' ---
    L_s = np.zeros((n_letters, N_s), dtype=float)
    L_s[0, :] = 1.0  # 'safe' everywhere

    # --- Lateral: lane-boundary–based labelling ---
    L_d = np.zeros((n_letters, N_d), dtype=float)
    for i, d in enumerate(centres_d):
        if abs(d) < lane_width:          # lanes 1 or 2 (drivable)
            L_d[0, i] = 1.0              # 'safe'
        else:                            # lanes 0 or 3 (boundary)
            L_d[2, i] = 1.0              # 'risk_high'

    return L_s, L_d


# ---------------------------------------------------------------------------
# Main builder  (public API)
# ---------------------------------------------------------------------------

def build_roundabout_abstraction(
    section_arc_length: float,
    lane_width: float,
    n_lanes: int = 4,
    dt: float = 0.05,
    N_s: int = 20,
    N_d: int = 16,
    v_s_max: float = 10.0,
    v_d_max: float = 2.0,
    n_speed_levels_s: int = 5,
    n_speed_levels_d: int = 5,
    k_speed: float = 0.5,
    k_lat: float = 0.1,
    road_reward: float = -1.0,
    edge_penalty: float = 50.0,
) -> Dict:
    """
    Build everything needed by ``SimpleDPSolver``.

    The lateral axis now spans all *n_lanes* lanes, centred at the
    boundary between lanes 1 and 2 (d = 0):

        d ∈ [−n_lanes/2 · lane_width,  +n_lanes/2 · lane_width]

    Returns
    -------
    dict with keys:
        P_s, P_d            – transition matrices
        centres_s, centres_d
        acc_s, acc_d
        state_cost_s, action_cost_s
        state_cost_d, action_cost_d
    """
    lateral_half = (n_lanes / 2.0) * lane_width   # e.g. 8.0 m for 4 lanes

    # ---- Longitudinal axis (wraps around the section) ----
    acc_s = np.linspace(0.0, v_s_max, n_speed_levels_s)   # forward speeds only

    P_s, centres_s = _build_1d_transitions(
        N=N_s, x_min=0.0, x_max=section_arc_length,
        dt=dt, speed_values=acc_s, wrap=True,
    )

    # ---- Lateral axis (all lanes, clamped, no wrap) ----
    acc_d = np.linspace(-v_d_max, v_d_max, n_speed_levels_d)

    P_d, centres_d = _build_1d_transitions(
        N=N_d, x_min=-lateral_half, x_max=lateral_half,
        dt=dt, speed_values=acc_d, wrap=False,
    )

    # ---- Cost maps ----
    state_cost_s, action_cost_s, state_cost_d, action_cost_d = _build_cost_maps(
        centres_s, centres_d, acc_s, acc_d, lane_width,
        n_lanes=n_lanes,
        k_speed=k_speed,
        k_lat=k_lat,
        road_reward=road_reward,
        edge_penalty=edge_penalty,
    )

    return {
        'P_s': P_s,
        'P_d': P_d,
        'centres_s': centres_s,
        'centres_d': centres_d,
        'acc_s': acc_s,
        'acc_d': acc_d,
        'state_cost_s': state_cost_s,
        'action_cost_s': action_cost_s,
        'state_cost_d': state_cost_d,
        'action_cost_d': action_cost_d,
    }


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def frenet_to_cell_indices(
    s: float, d: float,
    centres_s: np.ndarray,
    centres_d: np.ndarray,
) -> Tuple[int, int]:
    """Map continuous Frenet (s, d) to nearest grid cell indices."""
    cell_w_s = centres_s[1] - centres_s[0] if len(centres_s) > 1 else 1.0
    cell_w_d = centres_d[1] - centres_d[0] if len(centres_d) > 1 else 1.0

    i_s = int(np.clip(round((s - centres_s[0]) / cell_w_s), 0, len(centres_s) - 1))
    i_d = int(np.clip(round((d - centres_d[0]) / cell_w_d), 0, len(centres_d) - 1))
    return i_s, i_d


def cell_indices_to_frenet(
    i_s: int, i_d: int,
    centres_s: np.ndarray,
    centres_d: np.ndarray,
) -> Tuple[float, float]:
    """Map grid cell indices back to Frenet (s, d) coordinates."""
    return float(centres_s[i_s]), float(centres_d[i_d])


# ---------------------------------------------------------------------------
# Pedestrian Markov chain builder
# ---------------------------------------------------------------------------

def build_pedestrian_chain(
    N_p: int,
    lane_width: float,
    n_lanes: int = 4,
    p_move: float = 0.3,
    n_letters: int = 3,
    ped_on_road_penalty: float = 50.0,
) -> Dict:
    """
    Build a Markov-chain model for a pedestrian crossing from lane 3
    (outer boundary) toward lane 0 (inner island).

    The pedestrian's lateral position is discretised over the same span
    as the ego vehicle:  d_p ∈ [−N/2·w, +N/2·w].

    Transition
    ----------
    At each time step the pedestrian moves one cell *inward* (toward
    lower d / lane 0) with probability ``p_move``, or stays put with
    probability ``1 − p_move``.  The innermost cell is absorbing.

    Labels
    ------
    * Pedestrian on drivable lanes 1 or 2 (|d_p| < lane_width) → ``risk_high``
    * Pedestrian off-road (lanes 0 or 3)                        → ``safe``

    Parameters
    ----------
    N_p         : number of pedestrian grid cells.
    lane_width  : single lane width [m].
    n_lanes     : total number of concentric lanes (default 4).
    p_move      : per-step crossing probability (toward lane 0).
    n_letters   : should match DFA alphabet size (3).
    ped_on_road_penalty : state cost when pedestrian is on the road.

    Returns
    -------
    dict with keys:
        P_p       : (N_p, N_p)  stochastic transition matrix (1 action)
        centres_p : (N_p,)      cell centres
        L_p       : (n_letters, N_p)  label matrix
        cost_p    : (N_p,)      state cost (risk from pedestrian on road)
        rho_p     : (N_p,)      initial distribution (concentrated on lane 3)
    """
    lateral_half = (n_lanes / 2.0) * lane_width
    cell_w = 2.0 * lateral_half / N_p
    centres_p = np.linspace(-lateral_half + cell_w / 2.0,
                             lateral_half - cell_w / 2.0, N_p)

    # Transition matrix: move from high d (lane 3) toward low d (lane 0)
    # i.e. from high index toward low index.
    P_p = np.zeros((N_p, N_p), dtype=float)
    for i in range(N_p):
        if i > 0:
            P_p[i, i - 1] = p_move           # one cell toward lane 0
            P_p[i, i]     = 1.0 - p_move      # stay
        else:
            P_p[0, 0] = 1.0                   # absorbing at innermost cell

    # Label matrix — risk_high when pedestrian is on the road
    L_p = np.zeros((n_letters, N_p), dtype=float)
    for i, dp in enumerate(centres_p):
        if abs(dp) < lane_width:      # lanes 1 or 2 (on the road)
            L_p[2, i] = 1.0           # 'risk_high'
        else:                         # lanes 0 or 3 (off-road)
            L_p[0, i] = 1.0           # 'safe'

    # State cost: high when pedestrian on road
    cost_p = np.zeros(N_p, dtype=float)
    for i, dp in enumerate(centres_p):
        if abs(dp) < lane_width:
            cost_p[i] = ped_on_road_penalty

    # Initial distribution: pedestrian starts uniformly on lane 3
    rho_p = np.zeros(N_p, dtype=float)
    for i, dp in enumerate(centres_p):
        if dp >= lane_width:           # lane 3 (outer boundary)
            rho_p[i] = 1.0
    if rho_p.sum() > 0:
        rho_p /= rho_p.sum()
    else:
        rho_p[:] = 1.0 / N_p

    return {
        'P_p':       P_p,
        'centres_p': centres_p,
        'L_p':       L_p,
        'cost_p':    cost_p,
        'rho_p':     rho_p,
    }
