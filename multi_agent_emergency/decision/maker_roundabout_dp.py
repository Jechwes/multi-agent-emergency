"""
maker_roundabout_dp.py
======================
Decision maker for the roundabout scenario using the DFATree-based
dynamic programming algorithm from ``dfa_tree_r1_risk_min.py``.

Single-tree architecture
------------------------
A **single** DFATree is built offline using a reference lane's
abstraction.  The decoupled (s, d) policy is reused for every lane
because all sections share the same Frenet-frame dimensions.

The tree produces a risk-minimising policy that maps (s, d) to
(v_s, v_d) speed targets.  Lane changes are **not** part of the DP;
they are handled by the safety filter.

The DFATree structure is retained so that multi-agent / LTL extensions
can be added later without rearchitecting.
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

from decision.risk_LP.dfa_tree_r1_risk_min import DFATree
from abstraction.roundabout_abstraction import SysAbs1D


class RoundaboutDPDecisionMaker:
    """
    Offline DP solver + online policy lookup.

    Builds ONE DFATree from the abstraction data, then provides:
      - ``get_action(s, d)`` → (v_s, v_d)  policy lookup
      - ``get_value(s, d)``  → float         value function query

    Parameters
    ----------
    dfa       : RoundaboutDFA instance
    abs_data  : dict returned by ``build_abstraction()``
    gamma     : discount factor
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

        # Unpack abstraction
        self.acc_s = abs_data['acc_s']
        self.acc_d = abs_data['acc_d']
        self.centres_s = abs_data['centres_s']
        self.centres_d = abs_data['centres_d']

        # Build the single DFATree (offline)
        self.tree = self._build_tree(n_tree_iters, n_vi_per_iter, n_grow)

        # Current DFA state (for online tracking)
        self.q_current: int = dfa.S0

    # ------------------------------------------------------------------
    # Offline: build and solve the DFATree
    # ------------------------------------------------------------------

    def _build_tree(
        self,
        n_tree_iters: int,
        n_vi_per_iter: int,
        n_grow: int,
    ) -> DFATree:
        """Build and solve a single DFATree from the abstraction."""
        d = self.abs_data
        N_s = len(d['centres_s'])
        N_d = len(d['centres_d'])

        sysAbs = [SysAbs1D(d['P_s']), SysAbs1D(d['P_d'])]
        nx_list = [N_s, N_d]
        L = [d['L_s'], d['L_d']]
        cost_map = [d['state_cost_s'], d['state_cost_d']]
        action_cost_list = [d['action_cost_s'], d['action_cost_d']]
        rho = [
            np.ones(N_s, dtype=float) / N_s,
            np.ones(N_d, dtype=float) / N_d,
        ]

        # Add pedestrian dimension if available
        ped = d.get('ped_data')
        if ped is not None and 'P_p' in ped:
            N_p = len(ped['centres_p'])
            sysAbs.append(SysAbs1D(ped['P_p']))
            nx_list.append(N_p)
            L.append(d['L_p'])
            cost_map.append(d['cost_p'])
            action_cost_list.append(None)   # uncontrolled
            rho.append(ped['rho_p'])

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
    # Online: policy lookup  (no online solving)
    # ------------------------------------------------------------------

    def get_action(self, s: float, d: float) -> Tuple[float, float]:
        """
        Look up the offline policy at Frenet state (s, d).

        Returns (v_s, v_d) speed targets.
        """
        q = self.q_current
        if q == int(self.dfa.sink):
            return 0.0, 0.0

        i_s = self._to_index(s, self.centres_s)
        i_d = self._to_index(d, self.centres_d)

        pol_s = self.tree.pol[q][0]
        pol_d = self.tree.pol[q][1]

        if hasattr(pol_s, 'toarray'):
            pol_s = pol_s.toarray()
        if hasattr(pol_d, 'toarray'):
            pol_d = pol_d.toarray()

        a_s = int(np.argmax(pol_s[i_s, :]))
        a_d = int(np.argmax(pol_d[i_d, :]))

        return float(self.acc_s[a_s]), float(self.acc_d[a_d])

    def get_value(self, s: float, d: float) -> float:
        """
        Query the risk value at Frenet state (s, d).

            V_q(s, d) = sum_{n in L_Q^{-1}(q), n != root}  V_s[n, s] * V_d[n, d]

        The root is excluded because V[root, :] = 0 by initialisation and
        contributes nothing.  Lower value = safer.  Returns inf at sink.
        """
        q = self.q_current
        if q == int(self.dfa.sink):
            return float('inf')

        i_s = self._to_index(s, self.centres_s)
        i_d = self._to_index(d, self.centres_d)

        total = 0.0
        for n in self.tree.Q.get(q, []):
            if n == 0:          # root: V = 0, skip for clarity
                continue
            total += float(self.tree.V[0][n, i_s]) * float(self.tree.V[1][n, i_d])
        return total

    def update_dfa_state(self, label: str) -> int:
        """Advance the DFA state given the observed label."""
        self.q_current = self.dfa.next_state(self.q_current, label)
        return self.q_current

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_index(val: float, centres: np.ndarray) -> int:
        """Find nearest grid cell index."""
        return int(np.argmin(np.abs(centres - val)))
