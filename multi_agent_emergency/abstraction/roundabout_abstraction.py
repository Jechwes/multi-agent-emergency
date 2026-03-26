"""
roundabout_abstraction.py
=========================
Build a **single** decoupled 1-D MDP abstraction for the roundabout,
plus a lanelet connectivity graph.

Single-tree approach
--------------------
All lanes share the same grid dimensions:

    s in [0, L_section(ref_lane)]   longitudinal progress within a section
    d in [-w/2, +w/2]              lateral deviation from lane centre-line

Because the roundabout sections are identical in (s, d) sense, the
DFATree is computed **once** on a reference lane and the resulting
policy is reused for every lane.  Lane changes are discrete decisions
made by the safety filter, not part of the continuous DP.

Pedestrian dimension
--------------------
A third decoupled dimension models a crossing pedestrian as a Markov
chain.  The pedestrian's radial position is discretised over the full
roundabout width.

Cost design
-----------
*Higher cost = higher risk.*

**Longitudinal (s-axis)**
    state cost = 0  (all road, wrapping).
    action cost = -k_speed * v_s  (reward forward speed).

**Lateral (d-axis)**
    state cost = negative (reward) at centre, rising quadratically
      toward boundaries with a large positive penalty at the edges.
    action cost = k_lat * |v_d|  (discourage lateral wobble).
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

    Nodes are lane indices (0 .. n_lanes-1).  Edges represent possible
    lane changes between adjacent lanes.

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
        self.drivable_lanes = set(drivable_lanes)
        self.inner_radius = inner_radius
        self.lane_width = lane_width

    def is_drivable(self, lane_id: int) -> bool:
        return lane_id in self.drivable_lanes

    def adjacent_lanes(self, lane_id: int) -> List[int]:
        """Return all valid adjacent lane ids (drivable or not)."""
        result = []
        for dl in (-1, +1):
            adj = lane_id + dl
            if 0 <= adj < self.n_lanes:
                result.append(adj)
        return result



# ---------------------------------------------------------------------------
# 1-D transition matrix builder (stochastic)
# ---------------------------------------------------------------------------

def _distribute_transition_probability(
    P_flat: np.ndarray,
    i: int,
    a_idx: int,
    N: int,
    j_main: int,
    noise: float,
    wrap: bool,
) -> None:
    """
    Write one action-conditioned transition row with local process noise.

    Probability mass is split between the nominal successor cell and its
    immediate neighbors. This keeps transitions local and preserves
    stochastic consistency required by DP.
    """
    noise = float(np.clip(noise, 0.0, 0.49))
    p_main = 1.0 - 2.0 * noise
    col_offset = a_idx * N

    neighbors = [j_main - 1, j_main + 1]
    if wrap:
        neighbors = [n % N for n in neighbors]

    p_left = noise
    p_right = noise

    if not wrap:
        if neighbors[0] < 0:
            p_main += p_left
            p_left = 0.0
            neighbors[0] = j_main
        if neighbors[1] >= N:
            p_main += p_right
            p_right = 0.0
            neighbors[1] = j_main

    P_flat[i, col_offset + j_main] += p_main
    if p_left > 0.0:
        P_flat[i, col_offset + neighbors[0]] += p_left
    if p_right > 0.0:
        P_flat[i, col_offset + neighbors[1]] += p_right

def _build_1d_transitions(
    N: int,
    x_min: float,
    x_max: float,
    dt: float,
    speed_values: np.ndarray,
    process_noise: float = 0.08,
    wrap: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a stochastic transition matrix for a 1-D single-integrator.

    Returns
    -------
    P_flat  : (N, N*nu) action-conditioned transition matrix.
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
            _distribute_transition_probability(
                P_flat=P_flat,
                i=i,
                a_idx=a_idx,
                N=N,
                j_main=j,
                noise=process_noise,
                wrap=wrap,
            )

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
    boundary_penalty: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build cost arrays for s and d dimensions.

    Returns: state_cost_s, action_cost_s, state_cost_d, action_cost_d
    """
    half_w = lane_width / 2.0

    # Longitudinal
    state_cost_s = np.zeros(len(centres_s), dtype=float)
    action_cost_s = -k_speed * acc_s.copy()

    # Lateral: quadratic ramp from road_reward (centre) to boundary_penalty
    state_cost_d = np.empty(len(centres_d), dtype=float)
    for i, d in enumerate(centres_d):
        ratio = abs(d) / half_w if half_w > 0 else 0.0
        state_cost_d[i] = road_reward + (boundary_penalty - road_reward) * ratio ** 2
    action_cost_d = k_lat * np.abs(acc_d)

    return state_cost_s, action_cost_s, state_cost_d, action_cost_d


# ---------------------------------------------------------------------------
# Label-matrix builder
# ---------------------------------------------------------------------------

def build_label_matrices(
    N_s: int,
    N_d: int,
    centres_d: np.ndarray,
    lane_width: float,
    n_letters: int = 4,
    boundary_fraction: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-dimension label indicator matrices.

    Co-safety alphabet (4 letters):
        0 -> 'safe'          ¬nd ∧ ¬ped ∧ ¬car   (within the lane)
        1 -> 'non_drivable'  at/beyond the lane boundary
        2 -> 'pedestrian'    (not used in offline masks)
        3 -> 'other_car'     (not used in offline masks)

    Only letters 0 and 1 are assigned in the static offline masks.
    Letters 2, 3 are zero rows — they exist so that the matrix
    dimensions match the DFA alphabet size.  Because only letter 0
    self-loops in the co-safety DFA, the DFATree only ever reads
    the 'safe' row during value iteration.

    Returns: L_s (n_letters, N_s), L_d (n_letters, N_d)
    """
    half_w = lane_width / 2.0
    thresh = boundary_fraction * half_w

    L_s = np.zeros((n_letters, N_s), dtype=float)
    L_s[0, :] = 1.0  # all safe along s

    L_d = np.zeros((n_letters, N_d), dtype=float)
    for i, d in enumerate(centres_d):
        if abs(d) >= thresh:
            L_d[1, i] = 1.0   # non_drivable at edges
        else:
            L_d[0, i] = 1.0   # safe

    return L_s, L_d


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


# ---------------------------------------------------------------------------
# Main builder  (public API)
# ---------------------------------------------------------------------------

def build_abstraction(
    rmap,
    ref_lane: int = 2,
    drivable_lanes: Tuple[int, ...] = (1, 2),
    dt: float = 0.05,
    N_s: int = 10,
    N_d: int = 8,
    N_p: int = 0, #default to 0 for no pedestrian dimension
    v_s_max: float = 10.0,
    v_d_max: float = 2.0,
    n_speed_levels_s: int = 5,
    n_speed_levels_d: int = 5,
    k_speed: float = 0.5,
    k_lat: float = 0.1,
    road_reward: float = -1.0,
    boundary_penalty: float = 50.0,
    p_move: float = 0.3,
    process_noise_s: float = 0.08,
    process_noise_d: float = 0.08,
    ped_on_road_penalty: float = 50.0,
    n_letters: int = 4,
) -> Dict:
    """
    Build a single decoupled MDP abstraction + lanelet graph.

    One set of transition matrices (s, d, pedestrian) is built for
    ``ref_lane``.  The resulting DFATree policy is reused for all lanes.

    Returns
    -------
    dict with keys:
        P_s, P_d         : transition matrices
        centres_s/d      : cell centres
        L_s, L_d, L_p    : label matrices
        state_cost_s/d   : state costs
        action_cost_s/d  : action costs
        cost_p           : pedestrian state cost
        acc_s, acc_d     : speed action arrays
        arc_length       : reference lane section arc length
        ped_data         : dict with P_p, centres_p, rho_p
        graph            : LaneletGraph
    """
    lane_width = rmap.lane_width
    half_w = lane_width / 2.0
    n_lanes = rmap.n_lanes
    arc_length = rmap.section_arc_length(ref_lane)

    acc_s = np.linspace(0.0, v_s_max, n_speed_levels_s)
    acc_d = np.linspace(-v_d_max, v_d_max, n_speed_levels_d)

    # Transition matrices  (built once for ref_lane)
    P_s, centres_s = _build_1d_transitions(
        N=N_s, x_min=0.0, x_max=arc_length,
        dt=dt, speed_values=acc_s, process_noise=process_noise_s, wrap=True,
    )
    P_d, centres_d = _build_1d_transitions(
        N=N_d, x_min=-half_w, x_max=half_w,
        dt=dt, speed_values=acc_d, process_noise=process_noise_d, wrap=False,
    )

    # Labels
    L_s, L_d = build_label_matrices(
        N_s, N_d, centres_d, lane_width, n_letters=n_letters,
    )

    # Costs
    sc_s, ac_s, sc_d, ac_d = _build_cost_maps(
        centres_s, centres_d, acc_s, acc_d, lane_width,
        k_speed=k_speed, k_lat=k_lat,
        road_reward=road_reward, boundary_penalty=boundary_penalty,
    )

    # Pedestrian — only include if N_p >= 2
    ped_data = None
    L_p = None
    cost_p = None

    if N_p >= 2:
        ped_data = build_pedestrian_chain(N_p, lane_width, n_lanes, p_move)
        L_p = _build_pedestrian_labels(
            ped_data['centres_p'], lane_width, n_lanes, ref_lane, n_letters,
        )
        cost_p = _build_pedestrian_cost(
            ped_data['centres_p'], lane_width, n_lanes, ref_lane,
            penalty=ped_on_road_penalty,
        )

    # Lanelet graph (for lane change decisions by safety filter)
    graph = LaneletGraph(
        rmap.n_sections, n_lanes, list(drivable_lanes),
        rmap.inner_radius, lane_width,
    )

    return {
        'P_s': P_s, 'P_d': P_d,
        'centres_s': centres_s, 'centres_d': centres_d,
        'L_s': L_s, 'L_d': L_d, 'L_p': L_p,
        'state_cost_s': sc_s, 'action_cost_s': ac_s,
        'state_cost_d': sc_d, 'action_cost_d': ac_d,
        'cost_p': cost_p,
        'acc_s': acc_s, 'acc_d': acc_d,
        'arc_length': arc_length,
        'ped_data': ped_data,
        'graph': graph,
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


# ---------------------------------------------------------------------------
# Relative transition matrix builder
# ---------------------------------------------------------------------------

def _build_relative_transitions(
    N: int,
    dist_max: float,
    dt: float,
    speed_values: np.ndarray,
    process_noise: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build stochastic P_delta_s. Action v_s reduces obstacle distance."""
    nu = len(speed_values)
    cell_w = dist_max / N
    centres = np.linspace(cell_w / 2, dist_max - cell_w / 2, N)

    P_flat = np.zeros((N, N * nu), dtype=float)

    for a_idx, v in enumerate(speed_values):
        for i in range(N):
            # gap shrinks
            next_delta_s = centres[i] - v * dt
            # clip to avoid out of bounds (0 is collision, dist_max is safe)
            next_delta_s = np.clip(next_delta_s, 0.0, dist_max - 1e-6)
            j = int(np.clip(round((next_delta_s - cell_w / 2) / cell_w), 0, N - 1))
            _distribute_transition_probability(
                P_flat=P_flat,
                i=i,
                a_idx=a_idx,
                N=N,
                j_main=j,
                noise=process_noise,
                wrap=False,
            )

    return P_flat, centres

def build_relative_abstraction(
    rmap,
    dt: float = 0.05,
    N_s: int = 10,
    N_d: int = 8,
    dist_max: float = 60.0,
    v_s_max: float = 10.0,
    v_d_max: float = 2.0,
    n_speed_levels_s: int = 5,
    n_speed_levels_d: int = 5,
    k_speed: float = 0.5,
    k_lat: float = 0.1,
    collision_penalty: float = 50.0,
    boundary_penalty: float = 50.0,
    process_noise_s: float = 0.08,
    process_noise_d: float = 0.08,
    n_letters: int = 4,
) -> Dict:
    """Builds the relative-distance abstraction for the evasive policy."""
    lane_width = rmap.lane_width
    half_w = lane_width / 2.0
    
    acc_s = np.linspace(0.0, v_s_max, n_speed_levels_s)
    acc_d = np.linspace(-v_d_max, v_d_max, n_speed_levels_d)

    # Transition matrices
    P_delta_s, centres_delta_s = _build_relative_transitions(
        N=N_s,
        dist_max=dist_max,
        dt=dt,
        speed_values=acc_s,
        process_noise=process_noise_s,
    )
    P_d, centres_d = _build_1d_transitions(
        N=N_d,
        x_min=-half_w,
        x_max=half_w,
        dt=dt,
        speed_values=acc_d,
        process_noise=process_noise_d,
        wrap=False,
    )

    # Labels
    L_delta_s = np.zeros((n_letters, N_s), dtype=float)
    for i, ds in enumerate(centres_delta_s):
        if ds < 30.0:
            L_delta_s[2, i] = 1.0  # pedestrian label
        else:
            L_delta_s[0, i] = 1.0  # safe normally
    L_d = np.zeros((n_letters, N_d), dtype=float)
    thresh = 0.85 * half_w
    for i, d in enumerate(centres_d):
        if abs(d) >= thresh:
            L_d[1, i] = 1.0
        else:
            L_d[0, i] = 1.0

    # Costs
    state_cost_delta_s = np.zeros(N_s, dtype=float)
    for i, ds in enumerate(centres_delta_s):
        if ds < 30.0:
            state_cost_delta_s[i] = collision_penalty * (1.0 - ds/30.0)
    
    action_cost_delta_s = +k_speed * acc_s.copy()  # positive = penalize speed; higher v_s shrinks the gap to the pedestrian
    
    state_cost_d = np.zeros(N_d, dtype=float)
    for i, d in enumerate(centres_d):
        state_cost_d[i] = -1.0 + (boundary_penalty - (-1.0)) * (abs(d)/half_w)**2
    action_cost_d = k_lat * np.abs(acc_d)

    return {
        'P_s': P_delta_s,
        'P_d': P_d,
        'centres_s': centres_delta_s,
        'centres_d': centres_d,
        'L_s': L_delta_s,
        'L_d': L_d,
        'state_cost_s': state_cost_delta_s,
        'action_cost_s': action_cost_delta_s,
        'state_cost_d': state_cost_d,
        'action_cost_d': action_cost_d,
        'acc_s': acc_s,
        'acc_d': acc_d,
        'graph': LaneletGraph(rmap.n_sections, rmap.n_lanes, [], rmap.inner_radius, lane_width)
    }

