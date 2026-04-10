"""
maker_roundabout_dp.py
======================
Offline DP decision maker for the roundabout scenario using DFATree-based
risk-minimising value iteration (``dfa_tree_r1_risk_min.py``).

Single-dimension architecture
------------------------------
The DFATree is built offline over a **single dimension** (the ego
vehicle's lanelet graph) to enforce G(¬nd) — avoiding non-drivable
areas.

Since all vehicles share the same lanelet graph and the nd predicate
is per-dimension, the same policy is reused for the opponent vehicle.

Pedestrian avoidance and inter-vehicle collision avoidance are
cross-dimensional predicates that cannot be rank-1 decomposed.  They
are handled entirely by the online safety filter.

Online, ``get_action(...)`` returns a discrete action index per
controlled vehicle by looking up the offline policy at the current
lanelet indices.
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add FMTensJelmar to path
_FMTENS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..',
    'FMTensJelmar-python',
))
if _FMTENS_DIR not in sys.path:
    sys.path.insert(0, _FMTENS_DIR)

from decision.dfa_tree_r1_risk_min import DFATree
from abstraction.roundabout_abstraction import SysAbs1D


class RoundaboutDPDecisionMaker:
    """
    Offline DP solver + online policy lookup.

    Builds a single-dimension DFATree offline (ego vehicle only) to
    enforce G(¬nd).  The same policy is reused for the opponent.

    Parameters
    ----------
    dfa           : RoundaboutDFA instance
    abs_data      : dict with keys:
                      'P_ego'              : transition matrix
                      'L_ego'              : label matrix (2 letters)
                      'state_cost_ego'     : state cost vector
                      'action_cost_ego'    : action cost vector
                      'n_lanelets'         : number of lanelets
    gamma         : discount factor
    n_tree_iters, n_vi_per_iter, n_grow : DFATree solver parameters
    """

    def __init__(
        self,
        dfa,
        abs_data: Dict,
        gamma: float = 0.5,
        n_tree_iters: int = 3,
        n_vi_per_iter: int = 10,
        n_grow: int = 2,
    ) -> None:
        self.dfa = dfa
        self.abs_data = abs_data
        self.gamma = gamma

        self.n_lanelets = abs_data['n_lanelets']

        # Build the single-dimension DFATree (offline)
        print("[DP] Building tree...")
        self.tree = self._build_tree(
            abs_data, n_tree_iters, n_vi_per_iter, n_grow
        )

        # Current DFA state (for online tracking)
        self.q_current: int = dfa.S0

    # ------------------------------------------------------------------
    # Offline: build and solve the DFATree
    # ------------------------------------------------------------------

    def _build_tree(
        self,
        abs_data: Dict,
        n_tree_iters: int,
        n_vi_per_iter: int,
        n_grow: int,
    ) -> DFATree:
        """
        Build and solve a single-dimension DFATree (ego vehicle only).

        The tree enforces G(¬nd) over the ego vehicle's lanelet graph.
        The same policy is reused for the opponent vehicle online.
        """
        N = self.n_lanelets

        sysAbs = [SysAbs1D(abs_data['P_ego'])]
        nx_list = [N]
        L = [abs_data['L_ego']]
        cost_map = [abs_data['state_cost_ego']]
        action_cost_list = [abs_data['action_cost_ego']]
        rho = [np.ones(N, dtype=float) / N]

        # Single dimension: policy array is (n_dfa_states, 1)
        n_dfa_states = self.dfa.n_states
        pol_init = np.empty((n_dfa_states, 1), dtype=object)
        for q in range(n_dfa_states):
            pol_init[q][0] = None

        tree = DFATree(
            DFA=self.dfa,
            sysAbs=sysAbs,
            pol=pol_init,
            nx_list=nx_list,
            L=L,
            delta_VI=None,
            delta_pol=None,
            pol_mode="rt",
            VI_mode="rt",
            iter_idx=0,
            cost_map=cost_map,
        )
        tree.gamma = self.gamma
        tree.action_cost = action_cost_list
        tree.initiate()

        print(f"[DP] Solving DFATree (dims=1, "
              f"iters={n_tree_iters}, vi={n_vi_per_iter}, grow={n_grow})...")

        for it in range(n_tree_iters):
            for _ in range(n_grow):
                tree.grow()
            tree.set_iter(it)
            tree.maxpolicy(rho)
            for _ in range(n_vi_per_iter):
                tree.update_tree()

        print(f"[DP] Done ({tree.tree.number_of_nodes()} nodes)")
        return tree

    # ------------------------------------------------------------------
    # Online: policy lookup
    # ------------------------------------------------------------------

    def get_action(
        self,
        ego_lanelet: int,
        opp_lanelet: Optional[int] = None,
    ) -> Tuple[int, Optional[int]]:
        """
        Look up the offline policy at the current lanelet indices.

        The same single-dimension policy (nd-avoidance) is used for
        both ego and opponent vehicles.

        Parameters
        ----------
        ego_lanelet : flat lanelet index of ego vehicle.
        opp_lanelet : flat lanelet index of opponent vehicle, or ``None``
                      in single-agent mode.

        Returns
        -------
        (action_ego, action_opp)
            Discrete action indices (0=advance, 1=lane_in, 2=lane_out,
            3=yield).  ``action_opp`` is ``None`` in single-agent mode.
        """
        q = self.q_current
        if q == int(self.dfa.sink):
            yield_action = 3
            return yield_action, (yield_action if opp_lanelet is not None else None)

        action_ego = self._lookup_action(q, ego_lanelet)

        action_opp: Optional[int] = None
        if opp_lanelet is not None:
            action_opp = self._lookup_action(q, opp_lanelet)

        return action_ego, action_opp

    def _lookup_action(self, q: int, lanelet_idx: int) -> int:
        """Extract the greedy action from tree.pol[q][0] at lanelet_idx."""
        pol = self.tree.pol[q][0]
        if pol is None:
            return 3  # yield as fallback if policy not computed
        if hasattr(pol, 'toarray'):
            pol = pol.toarray()

        if lanelet_idx < 0 or lanelet_idx >= pol.shape[0]:
            print(f"[Warning] Lanelet index {lanelet_idx} out of bounds "
                  f"for policy shape {pol.shape}. Falling back to yield.")
            return 3

        return int(np.argmax(pol[lanelet_idx, :]))

    # ------------------------------------------------------------------
    # Online: value query
    # ------------------------------------------------------------------

    def get_value(
        self,
        lanelet_idx: int,
    ) -> float:
        """
        Risk value at a lanelet index from the single-dimension tree.

        Computes  Σ_n  V[0][n, lanelet_idx]  over tree nodes for the
        current DFA mode.

        Parameters
        ----------
        lanelet_idx : flat lanelet index.

        Returns
        -------
        float — total risk value (lower is safer).
        """
        q = self.q_current
        if q == int(self.dfa.sink):
            return float('inf')

        total = 0.0
        for n in self.tree.Q.get(q, []):
            total += float(self.tree.V[0][n, lanelet_idx])
        return total

    # ------------------------------------------------------------------
    # Online: DFA state tracking
    # ------------------------------------------------------------------

    def update_dfa_state(self, label: str) -> int:
        """Advance the DFA state given the observed label."""
        self.q_current = self.dfa.next_state(self.q_current, label)
        return self.q_current
