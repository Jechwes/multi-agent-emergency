"""
simple_dp.py
============
Simple decoupled 1-D value iteration for risk-minimizing roundabout control.

No DFA, no tree — just minimise cumulative cost over a grid MDP:

    V(s) = min_a  [ r_state(s') + r_action(a) + γ · V(s') ]

Cost structure
--------------
    **Longitudinal (s-axis):**
        - State cost:  0 for all cells (the s-axis wraps around the section)
        - Action cost: ``-k_speed · v_s(a)``   → reward forward motion
        ⟹ The policy always prefers higher forward speed.

    **Lateral (d-axis):**
        - State cost:  negative at lane centre, ramping sharply positive
          near the lane boundary → penalise leaving the drivable corridor.
        - Action cost: small penalty ``k_lat · |v_d(a)|`` → discourage wobble.
        ⟹ The policy favours staying centred.

Why this works
--------------
Higher cost ≡ higher risk.  Road cells carry *negative* cost (reward for
being on the road), while non-drivable / lane-edge regions carry large
*positive* cost.  Value iteration finds the policy that minimises the
infinite-horizon discounted sum of these costs, i.e. **minimises risk**.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List


class SimpleDPSolver:
    """
    Decoupled 1-D value iteration for risk minimisation.

    Solves two independent 1-D MDPs (longitudinal, lateral) offline.
    Querying the policy online is O(1) — just a table lookup.

    Parameters
    ----------
    P_s, P_d         : (N, N*nu) deterministic transition matrices.
    centres_s / d    : (N,) cell-centre coordinates.
    acc_s / d        : (nu,) discrete speed / action values.
    state_cost_s / d : (N,) per-cell state cost.
    action_cost_s    : (nu_s,) per-action cost (longitudinal).
    action_cost_d    : (nu_d,) per-action cost (lateral).
    gamma            : discount factor.
    n_iters          : maximum number of VI sweeps.
    """

    def __init__(
        self,
        P_s: np.ndarray,
        P_d: np.ndarray,
        centres_s: np.ndarray,
        centres_d: np.ndarray,
        acc_s: np.ndarray,
        acc_d: np.ndarray,
        state_cost_s: np.ndarray,
        state_cost_d: np.ndarray,
        action_cost_s: np.ndarray,
        action_cost_d: np.ndarray,
        gamma: float = 0.99,
        n_iters: int = 200,
    ) -> None:
        self.centres_s = np.asarray(centres_s, dtype=float)
        self.centres_d = np.asarray(centres_d, dtype=float)
        self.acc_s = np.asarray(acc_s, dtype=float)
        self.acc_d = np.asarray(acc_d, dtype=float)
        self.gamma = gamma

        N_s = len(centres_s)
        N_d = len(centres_d)
        nu_s = len(acc_s)
        nu_d = len(acc_d)

        print("[SimpleDPSolver] Running value iteration …")

        self.V_s, self.pol_s = self._value_iteration(
            P_s, state_cost_s, action_cost_s, N_s, nu_s, gamma, n_iters,
        )
        self.V_d, self.pol_d = self._value_iteration(
            P_d, state_cost_d, action_cost_d, N_d, nu_d, gamma, n_iters,
        )

        print(
            f"[SimpleDPSolver] Done.  "
            f"V_s ∈ [{self.V_s.min():.2f}, {self.V_s.max():.2f}]  "
            f"V_d ∈ [{self.V_d.min():.2f}, {self.V_d.max():.2f}]"
        )
        print(f"  pol_s actions → {self.acc_s[self.pol_s]}")
        print(f"  pol_d actions → {self.acc_d[self.pol_d]}")

    # ------------------------------------------------------------------
    #  Core: value iteration
    # ------------------------------------------------------------------

    @staticmethod
    def _value_iteration(
        P: np.ndarray,
        state_cost: np.ndarray,
        action_cost: np.ndarray,
        N: int,
        nu: int,
        gamma: float,
        n_iters: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standard Bellman value iteration on a 1-D MDP.

            Q(s, a) = r_action(a)  +  Σ_{s'} P(s'|s,a) · [r_state(s') + γ·V(s')]
            V(s)    = min_a Q(s, a)
        """
        V = np.zeros(N, dtype=float)

        for it in range(n_iters):
            target = state_cost + gamma * V           # (N,)
            Q = np.empty((N, nu), dtype=float)
            for a in range(nu):
                Pa = P[:, a * N : (a + 1) * N]       # (N, N)
                Q[:, a] = Pa @ target + action_cost[a]

            V_new = np.min(Q, axis=1)

            if np.max(np.abs(V_new - V)) < 1e-10:
                print(f"  VI converged after {it + 1} iterations")
                V = V_new
                break
            V = V_new

        # Recompute Q with final V for clean policy extraction
        target = state_cost + gamma * V
        Q = np.empty((N, nu), dtype=float)
        for a in range(nu):
            Pa = P[:, a * N : (a + 1) * N]
            Q[:, a] = Pa @ target + action_cost[a]

        policy = np.argmin(Q, axis=1)                 # (N,) int
        return V, policy

    # ------------------------------------------------------------------
    #  Online queries
    # ------------------------------------------------------------------

    def get_action(self, s: float, d: float) -> Tuple[float, float]:
        """Return optimal (v_s, v_d) for a Frenet state."""
        i_s = self._nearest(s, self.centres_s)
        i_d = self._nearest(d, self.centres_d)
        v_s = float(self.acc_s[self.pol_s[i_s]])
        v_d = float(self.acc_d[self.pol_d[i_d]])
        return v_s, v_d

    def get_value(self, s: float, d: float) -> Tuple[float, float]:
        """Return (V_s(s), V_d(d)) at a Frenet state."""
        i_s = self._nearest(s, self.centres_s)
        i_d = self._nearest(d, self.centres_d)
        return float(self.V_s[i_s]), float(self.V_d[i_d])

    # ------------------------------------------------------------------
    #  Rollout (for MPC reference trajectory)
    # ------------------------------------------------------------------

    def rollout_frenet(
        self,
        s0: float,
        d0: float,
        section_arc_length: float,
        dt: float,
        n_steps: int,
    ) -> List[Tuple[float, float, float, float, int]]:
        """
        Roll out the policy for *n_steps* at resolution *dt*.

        Returns a list of tuples::

            (s, d, v_s, v_d, section_offset)

        where ``section_offset`` is the number of whole-section wraps
        that occurred (0 means still in the starting section).
        """
        path: List[Tuple[float, float, float, float, int]] = []
        s, d = s0, d0
        sec_offset = 0

        for _ in range(n_steps):
            v_s, v_d = self.get_action(s, d)
            s += v_s * dt
            d += v_d * dt

            # wrap s around section
            while s >= section_arc_length:
                s -= section_arc_length
                sec_offset += 1
            while s < 0:
                s += section_arc_length
                sec_offset -= 1

            # clamp d
            d_max = self.centres_d[-1] + (self.centres_d[-1] - self.centres_d[-2]) / 2
            d_min = self.centres_d[0]  - (self.centres_d[1]  - self.centres_d[0])  / 2
            d = float(np.clip(d, d_min * 0.95, d_max * 0.95))

            path.append((s, d, v_s, v_d, sec_offset))

        return path

    # ------------------------------------------------------------------
    #  Private
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest(val: float, centres: np.ndarray) -> int:
        return int(np.argmin(np.abs(centres - val)))
