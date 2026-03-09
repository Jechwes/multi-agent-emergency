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
coordinates:

    s ∈ [0, L_section]    longitudinal progress within the current section
    d ∈ [-w/2, +w/2]      lateral deviation from lane centre-line

We discretise each axis independently into ``N_s`` and ``N_d`` grid cells
and build *deterministic* single-integrator transition matrices:

    P_s:  (N_s, N_s * nu_s)      longitudinal
    P_d:  (N_d, N_d * nu_d)      lateral

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

    • state cost = negative at the lane centre (reward for being on road),
      smoothly ramping to a large positive value near the lane edges
      (penalty for risk of leaving the drivable corridor).
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
    k_speed: float = 0.5,
    k_lat: float = 0.1,
    road_reward: float = -1.0,
    edge_penalty: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the risk-based cost arrays.

    Returns
    -------
    state_cost_s  : (N_s,)   – zero everywhere (all road).
    action_cost_s : (nu_s,)  – negative proportional to forward speed.
    state_cost_d  : (N_d,)   – negative at centre, positive at edges.
    action_cost_d : (nu_d,)  – small penalty for lateral speed magnitude.
    """
    N_s = len(centres_s)
    half_w = lane_width / 2.0

    # --- Longitudinal costs ---
    state_cost_s = np.zeros(N_s, dtype=float)
    # Reward higher speed (more negative → lower cost → preferred)
    action_cost_s = -k_speed * acc_s.copy()

    # --- Lateral costs ---
    # Smooth cost profile:
    #   road_reward at d=0, rising to +edge_penalty at |d|=half_w
    #   Using a quartic profile for a sharp rise near the edge.
    normalised = np.abs(centres_d) / half_w                       # 0 … 1
    state_cost_d = road_reward + (edge_penalty - road_reward) * normalised ** 4

    # Small penalty for lateral speed magnitude (discourages wobble)
    action_cost_d = k_lat * np.abs(acc_d)

    return state_cost_s, action_cost_s, state_cost_d, action_cost_d


# ---------------------------------------------------------------------------
# Label-matrix builder  (for DFATree)
# ---------------------------------------------------------------------------

def build_label_matrices(
    N_s: int,
    N_d: int,
    n_letters: int = 4,
    edge_cells: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-dimension label indicator matrices for the DFATree.

    Alphabet columns (matching ``RoundaboutDFA``):

        0 → 'r'  road (safe, not target)
        1 → 't'  target
        2 → 'n'  non-drivable
        3 → 'c'  collision

    Parameters
    ----------
    N_s         : number of longitudinal cells.
    N_d         : number of lateral cells.
    n_letters   : number of alphabet symbols (default 4).
    edge_cells  : how many cells on each lateral edge are non-drivable.

    Returns
    -------
    L_s : (n_letters, N_s)  – all road along s-axis.
    L_d : (n_letters, N_d)  – road in centre, non-drivable at edges.
    """
    # --- Longitudinal: entire section is drivable road ---
    L_s = np.zeros((n_letters, N_s), dtype=float)
    L_s[0, :] = 1.0  # 'r' everywhere

    # --- Lateral: centre is road, outermost cells are non-drivable ---
    L_d = np.zeros((n_letters, N_d), dtype=float)
    for i in range(N_d):
        if i < edge_cells or i >= N_d - edge_cells:
            L_d[2, i] = 1.0          # 'n' : non-drivable edge
        else:
            L_d[0, i] = 1.0          # 'r' : road

    return L_s, L_d


# ---------------------------------------------------------------------------
# Main builder  (public API)
# ---------------------------------------------------------------------------

def build_roundabout_abstraction(
    section_arc_length: float,
    lane_width: float,
    dt: float = 0.05,
    N_s: int = 20,
    N_d: int = 10,
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

    Returns
    -------
    dict with keys:
        P_s, P_d            – transition matrices
        centres_s, centres_d
        acc_s, acc_d
        state_cost_s, action_cost_s
        state_cost_d, action_cost_d
    """
    # ---- Longitudinal axis (wraps around the section) ----
    acc_s = np.linspace(0.0, v_s_max, n_speed_levels_s)   # forward speeds only

    P_s, centres_s = _build_1d_transitions(
        N=N_s, x_min=0.0, x_max=section_arc_length,
        dt=dt, speed_values=acc_s, wrap=True,
    )

    # ---- Lateral axis (clamped, no wrap) ----
    acc_d = np.linspace(-v_d_max, v_d_max, n_speed_levels_d)

    P_d, centres_d = _build_1d_transitions(
        N=N_d, x_min=-lane_width / 2.0, x_max=lane_width / 2.0,
        dt=dt, speed_values=acc_d, wrap=False,
    )

    # ---- Cost maps ----
    state_cost_s, action_cost_s, state_cost_d, action_cost_d = _build_cost_maps(
        centres_s, centres_d, acc_s, acc_d, lane_width,
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
