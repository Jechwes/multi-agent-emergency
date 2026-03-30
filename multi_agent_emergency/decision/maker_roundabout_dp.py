"""
maker_roundabout_dp.py
======================
Offline DP decision maker for the roundabout scenario using DFATree-based
risk-minimising value iteration (``dfa_tree_r1_risk_min.py``).

Single-tree architecture
------------------------
One DFATree is built offline at startup over all agent dimensions:

  - ego vehicle      : lanelet-graph transition matrix, controlled
  - opponent vehicle : lanelet-graph transition matrix, controlled
  - pedestrian       : Markov chain, uncontrolled

For single-agent mode (no opponent), only the ego + pedestrian
dimensions are included.

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

    Builds ONE DFATree offline over all agent dimensions, then provides:
      - ``get_action(...)``  → discrete action index per controlled vehicle
      - ``get_value(...)``   → factored risk value at the current joint state

    Parameters
    ----------
    dfa           : RoundaboutDFA instance
    abs_data      : dict with keys produced by the new lanelet-graph
                    abstraction builders:
                      'P_ego', 'P_opp'       : transition matrices (optional opp)
                      'L_ego', 'L_opp'       : label matrices
                      'state_cost_ego/opp'   : state cost vectors
                      'action_cost_ego/opp'  : action cost vectors
                      'n_lanelets'           : number of lanelets
                      'ped_data'             : dict with P_p, centres_p, rho_p (optional)
                      'L_p', 'cost_p'        : pedestrian labels/cost (optional)
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
        self.has_opponent = 'P_opp' in abs_data
        self.has_pedestrian = (abs_data.get('ped_data') is not None
                               and abs_data.get('L_p') is not None)

        # Dimension index bookkeeping
        self.dim_ego = 0
        self.dim_opp: Optional[int] = None
        self.dim_ped: Optional[int] = None

        next_dim = 1
        if self.has_opponent:
            self.dim_opp = next_dim
            next_dim += 1
        if self.has_pedestrian:
            self.dim_ped = next_dim

        # Build the single DFATree (offline)
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
        Assemble dimensions and solve a single DFATree.

        Dimensions are added in order:
          0 : ego vehicle      (controlled)
          1 : opponent vehicle (controlled, only if present)
          ? : pedestrian       (uncontrolled, only if present)

        Single-agent mode omits the opponent dimension entirely.
        """
        N = self.n_lanelets

        # -- always: ego dimension --
        sysAbs = [SysAbs1D(abs_data['P_ego'])]
        nx_list = [N]
        L = [abs_data['L_ego']]
        cost_map = [abs_data['state_cost_ego']]
        action_cost_list = [abs_data['action_cost_ego']]
        rho = [np.ones(N, dtype=float) / N]

        # -- optional: opponent dimension --
        if self.has_opponent:
            sysAbs.append(SysAbs1D(abs_data['P_opp']))
            nx_list.append(N)
            L.append(abs_data['L_opp'])
            cost_map.append(abs_data['state_cost_opp'])
            action_cost_list.append(abs_data['action_cost_opp'])
            rho.append(np.ones(N, dtype=float) / N)

        # -- optional: pedestrian dimension (uncontrolled) --
        if self.has_pedestrian:
            ped = abs_data['ped_data']
            N_p = len(ped['centres_p'])
            sysAbs.append(SysAbs1D(ped['P_p']))
            nx_list.append(N_p)
            L.append(abs_data['L_p'])
            cost_map.append(abs_data['cost_p'])
            action_cost_list.append(None)   # uncontrolled
            rho.append(ped['rho_p'])

        # -- initialise policy array --
        n_dims = len(sysAbs)
        n_dfa_states = self.dfa.n_states
        pol_init = np.empty((n_dfa_states, n_dims), dtype=object)
        for q in range(n_dfa_states):
            for dim in range(n_dims):
                pol_init[q][dim] = None

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

        print(f"[DP] Solving DFATree (dims={n_dims}, "
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
            # In the failure state, yield for all vehicles
            yield_action = 3
            return yield_action, (yield_action if self.has_opponent else None)

        # -- ego action --
        action_ego = self._lookup_action(q, self.dim_ego, ego_lanelet)

        # -- opponent action (if present) --
        action_opp: Optional[int] = None
        if self.has_opponent and opp_lanelet is not None:
            action_opp = self._lookup_action(q, self.dim_opp, opp_lanelet)

        return action_ego, action_opp

    def _lookup_action(self, q: int, dim: int, lanelet_idx: int) -> int:
        """Extract the greedy action from tree.pol[q][dim] at lanelet_idx."""
        pol = self.tree.pol[q][dim]
        if pol is None:
            return 3  # yield as fallback if policy not computed
        if hasattr(pol, 'toarray'):
            pol = pol.toarray()
        
        if lanelet_idx < 0 or lanelet_idx >= pol.shape[0]:
            print(f"[Warning] Lanelet index {lanelet_idx} out of bounds for policy shape {pol.shape}. Falling back to yield.")
            return 3
            
        return int(np.argmax(pol[lanelet_idx, :]))

    # ------------------------------------------------------------------
    # Online: value query
    # ------------------------------------------------------------------

    def get_value(
        self,
        ego_lanelet: int,
        opp_lanelet: Optional[int] = None,
        ped_state: Optional[int] = None,
    ) -> float:
        """
        Factored risk value at the current joint state.

        Computes  Σ_n  ∏_d  V[d][n, idx_d]  over tree nodes and all
        active dimensions (ego × opponent × pedestrian).

        Parameters
        ----------
        ego_lanelet : flat lanelet index of ego vehicle.
        opp_lanelet : flat lanelet index of opponent, or ``None``.
        ped_state   : pedestrian grid index, or ``None``.

        Returns
        -------
        float — total risk value (lower is safer).
        """
        q = self.q_current
        if q == int(self.dfa.sink):
            return float('inf')

        # Build list of (dimension_index, state_index) pairs
        dims_and_indices: List[Tuple[int, int]] = [(self.dim_ego, ego_lanelet)]

        if self.has_opponent and opp_lanelet is not None:
            dims_and_indices.append((self.dim_opp, opp_lanelet))

        if self.has_pedestrian and ped_state is not None:
            dims_and_indices.append((self.dim_ped, ped_state))

        total = 0.0
        for n in self.tree.Q.get(q, []):
            product = 1.0
            for dim, idx in dims_and_indices:
                product *= float(self.tree.V[dim][n, idx])
            total += product
        return total

    # ------------------------------------------------------------------
    # Online: DFA state tracking
    # ------------------------------------------------------------------

    def update_dfa_state(self, label: str) -> int:
        """Advance the DFA state given the observed label."""
        self.q_current = self.dfa.next_state(self.q_current, label)
        return self.q_current
