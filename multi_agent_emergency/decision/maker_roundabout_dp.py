"""
maker_roundabout_dp.py
======================
Decision maker for the roundabout scenario using the DFATree-based
dynamic programming algorithm from ``dfa_tree_r1_risk_min.py``.

This replaces the old LP-based ``Risk_LTL`` decision maker with the
decoupled DP approach:

    V_{q,k+1} = Σ_{(α_i, q') ∈ N_q}  T_α^{π_q}[N-1-k] (V^π_{q',k})

where T_α^{π_q}(V)(s) = E^{s'}[ L_α(s') · (V(s') + c(s')) | s, a = π_q(s) ]

and c(s) is the cost map (risk penalty for lane deviation, etc.).

The decoupled dynamics means:
    V(s, d, p) ≈ V_s(s) · V_d(d) · V_p(p)

where V_s, V_d, and V_p are computed independently by the DFATree over
the longitudinal, lateral, and pedestrian axes respectively.
The pedestrian axis is *uncontrolled* (nu_p = 1): the ego vehicle cannot
influence its movement; the Markov chain propagation happens automatically
inside the DFATree value update.

Safety-filter value function
----------------------------
Because the safety DFA ``G(safe)`` creates a degenerate tree (all nodes
at q_safe with label "safe", so the label‐mask kills the positive risk
costs), the DFATree V stays at zero everywhere.

A **standalone risk value function** is therefore computed separately via
standard Bellman iteration:

    V_risk_d(x) = risk_d(x) + γ · Σ_{x'} P(x'|x, π_d) · V_risk_d(x')

where risk_d(x) = Σ_l L[d][l,x] · risk_cost_dfa[l].

This produces meaningful risk values: high for unsafe cells, elevated
near boundaries, and propagated through stochastic transitions
(especially for the pedestrian Markov chain).

The ``SafetyFilter`` class monitors this value function every tick and
intervenes (brakes) when the combined risk exceeds a configurable
threshold.
"""

from __future__ import annotations

import sys
import os
import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

# Add FMTensJelmar to path so we can import if needed
_FMTENS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..',
    'FMTensJelmar-python',
))
if _FMTENS_DIR not in sys.path:
    sys.path.insert(0, _FMTENS_DIR)

# Import DFATree from the local decision package
from decision.risk_LP.dfa_tree_r1_risk_min import DFATree
from decision.abstraction.utils.pc_utils import Pc


class RoundaboutDPDecisionMaker:
    """
    High-level decision maker that runs the DFATree DP algorithm and
    returns Frenet-frame targets for the MPC controller.

    Supports D = 2 (s, d) or D = 3 (s, d, pedestrian) dimensions.
    When D = 3 the pedestrian dimension is *uncontrolled*: it has a
    single-action transition matrix (pure Markov chain), so the
    returned ego action is always (v_s, v_d).

    Parameters
    ----------
    dfa         : RoundaboutDFA instance
    sysAbs      : list of SysAbs1D  [longitudinal, lateral (, pedestrian)]
    nx_list     : [N_s, N_d (, N_p)]
    L           : [L_s, L_d (, L_p)] label matrices
    pol         : initial policy array
    rho         : initial state distribution per dimension
    cost_map    : [cost_s, cost_d (, cost_p)] per-cell cost arrays
    centres_s   : (N_s,) longitudinal grid centres
    centres_d   : (N_d,) lateral grid centres
    centres_p   : (N_p,) pedestrian grid centres (optional, only for D=3)
    acc_s       : longitudinal speed/action values
    acc_d       : lateral speed/action values
    n_tree_iters     : number of tree grow + VI + policy iterations
    n_vi_per_iter    : number of value-iteration sweeps per tree iteration
    n_grow           : number of tree expansions before VI
    gamma            : discount factor
    """

    def __init__(
        self,
        dfa,
        sysAbs: list,
        nx_list: list,
        L: list,
        pol,
        rho: list,
        cost_map: list,
        centres_s: np.ndarray,
        centres_d: np.ndarray,
        acc_s: np.ndarray,
        acc_d: np.ndarray,
        n_tree_iters: int = 5,
        n_vi_per_iter: int = 10,
        n_grow: int = 3,
        gamma: float = 0.99,
        action_cost: Optional[list] = None,
        centres_p: Optional[np.ndarray] = None,
    ) -> None:
        self.dfa = dfa
        self.sysAbs = sysAbs
        self.nx_list = nx_list
        self.L = L
        self.pol = pol
        self.rho = rho
        self.cost_map = cost_map
        self.centres_s = centres_s
        self.centres_d = centres_d
        self.centres_p = centres_p           # optional pedestrian grid
        self.acc_s = acc_s
        self.acc_d = acc_d
        self.action_cost = action_cost       # [action_cost_s, action_cost_d(, None)]

        self.n_dims = len(sysAbs)            # 2 or 3
        self.n_tree_iters = n_tree_iters
        self.n_vi_per_iter = n_vi_per_iter
        self.n_grow = n_grow
        self.gamma = gamma

        # Current DFA state (tracked across simulation steps)
        self.q_current: int = dfa.S0

        # Build and solve the DFA tree offline
        self.tree: Optional[DFATree] = None
        self._build_tree()

    def _build_tree(self) -> None:
        """Build and solve the DFA tree once (offline phase)."""
        print("[RoundaboutDP] Building DFA tree...")

        self.tree = DFATree(
            DFA=self.dfa,
            sysAbs=self.sysAbs,
            pol=self.pol,
            nx_list=self.nx_list,
            L=self.L,
            delta_VI=None,       # no robustness margins for now
            delta_pol=None,
            pol_mode="rt",
            VI_mode="rt",
            iter_idx=0,
            cost_map=self.cost_map,
        )
        self.tree.gamma = self.gamma

        # Inject per-action costs so that maxpolicy prefers low-cost actions
        # (e.g. forward speed is rewarded via negative action cost)
        if self.action_cost is not None:
            self.tree.action_cost = self.action_cost

        self.tree.initiate()

        for it in range(self.n_tree_iters):
            print(f"[RoundaboutDP] Iteration {it + 1}/{self.n_tree_iters}")

            # Grow tree
            for _ in range(self.n_grow):
                self.tree.grow()

            # Policy improvement → builds Pxx
            self.tree.set_iter(it)
            self.tree.maxpolicy(self.rho)

            # Value iteration sweeps
            for _ in range(self.n_vi_per_iter):
                self.tree.update_tree()

        print(f"[RoundaboutDP] Tree built with {self.tree.tree.number_of_nodes()} nodes, D={self.n_dims}")

        # --- Compute the standalone risk value function ---
        self.compute_risk_field(n_iters=80, gamma=self.gamma)

    # ------------------------------------------------------------------
    # Online: extract action from solved value function
    # ------------------------------------------------------------------

    def get_action(
        self,
        s: float,
        d: float,
        current_section: int,
        current_lane: int,
    ) -> Tuple[float, float, int]:
        """
        Extract the best (speed_s, speed_d) action for the current Frenet state.

        Returns
        -------
        v_s       : longitudinal speed target [m/s]
        v_d       : lateral speed target [m/s]
        q_next    : next DFA state (for tracking)
        """
        # Map continuous state to grid indices
        i_s = self._to_index(s, self.centres_s)
        i_d = self._to_index(d, self.centres_d)

        # Find the best actions from the solved policy
        q = self.q_current

        # Only skip the absorbing sink state; the accepting state (F)
        # is the normal operating state in the safety-filter DFA.
        if q == int(self.dfa.sink):
            # Trapped in failure: brake / maintain course
            return float(self.acc_s[len(self.acc_s) // 2]), 0.0, q

        # Get policy for current DFA state
        pol_s = self.tree.pol[q][0]  # (N_s, nu_s)
        pol_d = self.tree.pol[q][1]  # (N_d, nu_d)

        # Extract action from sparse policy
        if hasattr(pol_s, 'toarray'):
            pol_s = pol_s.toarray()
        if hasattr(pol_d, 'toarray'):
            pol_d = pol_d.toarray()

        a_s_idx = int(np.argmax(pol_s[i_s, :]))
        a_d_idx = int(np.argmax(pol_d[i_d, :]))

        v_s = float(self.acc_s[a_s_idx])
        v_d = float(self.acc_d[a_d_idx])

        return v_s, v_d, q

    def update_dfa_state(self, label: str) -> int:
        """
        Advance the DFA state given the observed label at the new cell.

        Parameters
        ----------
        label : one of 'safe', 'risk_low', 'risk_high'

        Returns
        -------
        New DFA state
        """
        self.q_current = self.dfa.next_state(self.q_current, label)
        return self.q_current

    def get_value_at(self, s: float, d: float, p: Optional[float] = None) -> float:
        """
        Get the product value V_s(s) · V_d(d) [· V_p(p)] at a Frenet point,
        minimised (lowest risk) over all relevant tree nodes.
        """
        i_s = self._to_index(s, self.centres_s)
        i_d = self._to_index(d, self.centres_d)
        i_p = self._to_index(p, self.centres_p) if (p is not None and self.centres_p is not None) else None

        q = self.q_current

        # Sink state → infinite risk
        if q == int(self.dfa.sink):
            return float('inf')

        best = float('inf')
        for n in self.tree.Q.get(q, []):
            v_s = float(self.tree.V[0][n, i_s])
            v_d = float(self.tree.V[1][n, i_d])
            prod = v_s * v_d
            if self.n_dims >= 3 and i_p is not None:
                v_p = float(self.tree.V[2][n, i_p])
                prod *= v_p
            if prod < best:
                best = prod
        return best

    def get_optimal_path_frenet(
        self,
        s: float,
        d: float,
        horizon: int = 10,
    ) -> List[Tuple[float, float]]:
        """
        Roll out the policy for `horizon` steps to get a preview path
        in Frenet coordinates (s, d).
        """
        path = [(s, d)]
        curr_s, curr_d = s, d

        for _ in range(horizon):
            i_s = self._to_index(curr_s, self.centres_s)
            i_d = self._to_index(curr_d, self.centres_d)

            q = self.q_current

            # Only stop rolling out if we're in the failure sink
            if q == int(self.dfa.sink):
                break

            pol_s = self.tree.pol[q][0]
            pol_d = self.tree.pol[q][1]

            if hasattr(pol_s, 'toarray'):
                pol_s = pol_s.toarray()
            if hasattr(pol_d, 'toarray'):
                pol_d = pol_d.toarray()

            a_s_idx = int(np.argmax(pol_s[i_s, :]))
            a_d_idx = int(np.argmax(pol_d[i_d, :]))

            v_s = float(self.acc_s[a_s_idx])
            v_d = float(self.acc_d[a_d_idx])

            dt = 0.5  # match the abstraction dt
            curr_s += v_s * dt
            curr_d += v_d * dt

            # Clip
            curr_s = float(np.clip(curr_s, self.centres_s[0], self.centres_s[-1]))
            curr_d = float(np.clip(curr_d, self.centres_d[0], self.centres_d[-1]))

            path.append((curr_s, curr_d))

        return path

    # ------------------------------------------------------------------
    # Risk value function (standalone Bellman, bypasses DFATree V)
    # ------------------------------------------------------------------

    def compute_risk_field(
        self,
        n_iters: int = 50,
        gamma: Optional[float] = None,
        gamma_risk: Optional[float] = None,
    ) -> None:
        """
        Compute per-dimension risk value functions using standard Bellman
        iteration with risk costs derived from the DFA label matrices.

        For each dimension *d*:
            risk_d(x) = Σ_l L[d][l, x] · risk_cost_dfa[l]

        Ego-controlled dimensions (s, d) use **instantaneous** risk only
        (no Bellman propagation), since the ego can change its position.

        The uncontrolled pedestrian dimension uses Bellman iteration with
        a separate ``gamma_risk`` discount to predict near-term risk:
            V_risk_p  ← risk_p + γ_risk · P_p @ V_risk_p

        Parameters
        ----------
        n_iters    : max Bellman iterations for the pedestrian dimension
        gamma      : unused (kept for backwards compatibility)
        gamma_risk : discount factor for pedestrian risk Bellman
                     (default: 0.5 for short-horizon prediction)
        """
        if gamma_risk is None:
            gamma_risk = 0.4  # moderate horizon → react when ped is close

        # --- Map DFA letter costs to a numeric vector ---
        rc = np.array([
            self.dfa.risk_cost.get('safe', 0.0),
            self.dfa.risk_cost.get('risk_low', 5.0),
            self.dfa.risk_cost.get('risk_high', 50.0),
        ], dtype=float)

        self.V_risk: list = []  # one (N_d,) array per dimension

        q = int(self.dfa.F)   # operating DFA state

        for d in range(self.n_dims):
            N = self.nx_list[d]
            n_letters = self.L[d].shape[0]

            # Per-cell instantaneous risk: risk(x) = Σ_l L[l,x] · rc[l]
            risk_d = np.zeros(N, dtype=float)
            for l in range(min(n_letters, len(rc))):
                risk_d += self.L[d][l, :].astype(float) * rc[l]

            # Ego-controlled dims (0=s, 1=d): instantaneous risk only.
            # Only the uncontrolled pedestrian dim (2) gets Bellman.
            if d < 2:
                self.V_risk.append(risk_d.copy())
                continue

            # --- Bellman iteration for uncontrolled dim (pedestrian) ---
            Pxx_d = self.tree.Pxx[q][d]
            if Pxx_d is None:
                self.V_risk.append(risk_d.copy())
                continue

            # Get a dense (N, N) matrix from Pxx_d
            if hasattr(Pxx_d, 'toarray'):
                P = Pxx_d.toarray().astype(float)
            elif hasattr(Pxx_d, 'stoch'):
                P = np.asarray(Pxx_d.stoch, dtype=float)
            else:
                P = np.asarray(Pxx_d, dtype=float)

            # If P is (N, N*nu) with nu=1, just take first N columns
            if P.ndim == 2 and P.shape[1] != N and P.shape[1] % N == 0:
                nu = P.shape[1] // N
                if nu == 1:
                    pass  # shape is already (N, N)
                else:
                    # Uncontrolled dim should have nu=1, but handle gracefully
                    P = P[:, :N]

            # --- Bellman iteration with short-horizon gamma_risk ---
            V = risk_d.copy()
            for _ in range(n_iters):
                V_new = risk_d + gamma_risk * (P @ V)
                if np.max(np.abs(V_new - V)) < 1e-6:
                    break
                V = V_new

            self.V_risk.append(V)

        print(f"[RoundaboutDP] Risk field computed "
              f"(ped Bellman: {n_iters} iters, γ_risk={gamma_risk:.2f})")
        for d in range(self.n_dims):
            v = self.V_risk[d]
            print(f"  dim {d}: min={v.min():.2f}  max={v.max():.2f}  "
                  f"nonzero={np.count_nonzero(v)}/{len(v)}")

    def get_risk_at(
        self,
        s: float,
        d: float,
        p: Optional[float] = None,
    ) -> float:
        """
        Query the risk value function at a continuous Frenet state.

        Returns the **sum** of per-dimension risks (additive combination):
            risk(s, d, p) = V_risk_s(s) + V_risk_d(d) + V_risk_p(p)

        Higher values indicate more dangerous states.
        """
        if not hasattr(self, 'V_risk') or self.V_risk is None:
            return 0.0

        i_s = self._to_index(s, self.centres_s)
        i_d = self._to_index(d, self.centres_d)

        risk = float(self.V_risk[0][i_s]) + float(self.V_risk[1][i_d])

        if self.n_dims >= 3 and p is not None and self.centres_p is not None:
            i_p = self._to_index(p, self.centres_p)
            risk += float(self.V_risk[2][i_p])

        return risk

    def get_risk_components(
        self,
        s: float,
        d: float,
        p: Optional[float] = None,
    ) -> dict:
        """
        Return per-dimension risk breakdown for diagnostics.
        """
        if not hasattr(self, 'V_risk') or self.V_risk is None:
            return {'s': 0.0, 'd': 0.0, 'p': 0.0, 'total': 0.0}

        i_s = self._to_index(s, self.centres_s)
        i_d = self._to_index(d, self.centres_d)
        r_s = float(self.V_risk[0][i_s])
        r_d = float(self.V_risk[1][i_d])
        r_p = 0.0

        if self.n_dims >= 3 and p is not None and self.centres_p is not None:
            i_p = self._to_index(p, self.centres_p)
            r_p = float(self.V_risk[2][i_p])

        return {'s': r_s, 'd': r_d, 'p': r_p, 'total': r_s + r_d + r_p}

    def get_safest_lane_d(self, p: Optional[float] = None) -> float:
        """
        Return the lateral position (d) that has the lowest combined
        lateral + pedestrian risk.  Used for evasive lane switching.
        """
        if not hasattr(self, 'V_risk') or self.V_risk is None:
            return 0.0

        combined = self.V_risk[1].copy()  # lateral risk per cell
        if self.n_dims >= 3 and p is not None:
            i_p = self._to_index(p, self.centres_p)
            combined = combined + float(self.V_risk[2][i_p])

        best_idx = int(np.argmin(combined))
        return float(self.centres_d[best_idx])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_index(val: float, centres: np.ndarray) -> int:
        """Find nearest grid cell index."""
        return int(np.argmin(np.abs(centres - val)))
