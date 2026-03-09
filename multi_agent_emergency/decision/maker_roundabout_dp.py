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
    V(s, d) ≈ V_s(s) · V_d(d)

where V_s and V_d are computed independently by the DFATree over the
longitudinal and lateral axes respectively.
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

    Parameters
    ----------
    dfa         : RoundaboutDFA instance
    sysAbs      : list of SysAbs1D [longitudinal, lateral]
    nx_list     : [N_s, N_d]
    L           : [L_s, L_d] label matrices
    pol         : initial policy array
    rho         : initial state distribution per dimension
    cost_map    : [cost_s, cost_d] per-cell cost arrays
    centres_s   : (N_s,) longitudinal grid centres
    centres_d   : (N_d,) lateral grid centres
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
        self.acc_s = acc_s
        self.acc_d = acc_d
        self.action_cost = action_cost   # [action_cost_s, action_cost_d]

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

        print(f"[RoundaboutDP] Tree built with {self.tree.tree.number_of_nodes()} nodes")

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
        skip = {int(self.dfa.F), int(self.dfa.sink)}

        if q in skip:
            # Accepting or trapped: just maintain course
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
        label : one of 'r', 't', 'n', 'c'

        Returns
        -------
        New DFA state
        """
        self.q_current = self.dfa.next_state(self.q_current, label)
        return self.q_current

    def get_value_at(self, s: float, d: float) -> float:
        """
        Get the product value V_s(s) · V_d(d) at a Frenet point,
        maximised over all relevant tree nodes.
        """
        i_s = self._to_index(s, self.centres_s)
        i_d = self._to_index(d, self.centres_d)

        q = self.q_current
        skip = {int(self.dfa.F), int(self.dfa.sink)}

        if q in skip:
            return 1.0 if self.dfa.is_accepting(q) else 0.0

        best = 0.0
        for n in self.tree.Q.get(q, []):
            v_s = float(self.tree.V[0][n, i_s])
            v_d = float(self.tree.V[1][n, i_d])
            prod = v_s * v_d
            if prod > best:
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
            skip = {int(self.dfa.F), int(self.dfa.sink)}

            if q in skip:
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
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_index(val: float, centres: np.ndarray) -> int:
        """Find nearest grid cell index."""
        return int(np.argmin(np.abs(centres - val)))
