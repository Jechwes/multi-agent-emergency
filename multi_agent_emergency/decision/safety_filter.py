"""
safety_filter.py
================
Online safety filter for the roundabout scenario.

At every tick the filter evaluates the **predictive horizon risk** by
scoring each future cell along the MPC look-ahead with ``stage_risk_cost()``
and summing with a discount factor gamma:

    R = Σ_k  γ^k · cost(cell_k)

The accumulated risk R is compared against two thresholds to select a
mode:

    NOMINAL  – R ≤ soft_threshold   → use nominal DP action
    CAUTION  – soft < R ≤ hard      → slow down (reduce cruise speed)
    BRAKE    – R > hard_threshold   → emergency yield (full stop)

A higher threshold tolerates more risk and lets the car brake closer
to the obstacle.  A lower threshold makes it react sooner.

``suggest_lane()`` returns the safest adjacent drivable lane when the car
is in CAUTION mode.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np


# ── Risk level enum ──────────────────────────────────────────────────
class RiskLevel(Enum):
    NOMINAL  = auto()   # safe, no intervention
    CAUTION  = auto()   # elevated risk, slow down
    BRAKE    = auto()   # high risk, emergency stop


# ── SafetyFilter ─────────────────────────────────────────────────────
class SafetyFilter:
    """
    Online safety filter using accumulated discounted risk.

    Parameters
    ----------
    dfa              : RoundaboutDFA (provides get_risk_cost per label)
    graph            : LaneletGraph  (lane connectivity)
    n_lanes          : total number of lanes
    drivable_lanes   : set of normally-drivable lane indices
    caution_speed_factor : cruise speed is multiplied by this in CAUTION
    """

    def __init__(
        self,
        dfa,
        graph,
        n_lanes: int,
        drivable_lanes: set,
        caution_speed_factor: float = 0.4,
    ) -> None:
        self.dfa = dfa
        self.graph = graph
        self.n_lanes = n_lanes
        self.drivable_lanes = set(drivable_lanes)
        self.caution_speed_factor = caution_speed_factor

    # ------------------------------------------------------------------
    # Per-cell risk cost
    # ------------------------------------------------------------------

    def stage_risk_cost(
        self,
        ego_section: int,
        ego_lane: int,
        n_sections: int,
        ped_section: Optional[int] = None,
        ped_lane: Optional[int] = None,
        opp_section: Optional[int] = None,
        opp_lane: Optional[int] = None,
    ) -> Tuple[float, str]:
        """
        Risk cost for a single cell based on what occupies it.

        Returns the DFA risk cost for the worst hazard present in the
        cell, or 0 if the cell is clear.  No distance cutoffs — the
        caller accumulates costs over the horizon with gamma discount,
        and the thresholds determine the reaction.
        """
        # Non-drivable lane
        if ego_lane not in self.drivable_lanes:
            return float(self.dfa.get_risk_cost('non_drivable')), 'non_drivable'

        # Pedestrian occupies same section
        if (
            ped_section is not None
            and ped_lane is not None
            and ped_lane in self.drivable_lanes
            and ped_section == ego_section
        ):
            return float(self.dfa.get_risk_cost('pedestrian')), 'pedestrian'

        # Opponent occupies same section and lane
        if (
            opp_section is not None
            and opp_lane is not None
            and opp_section == ego_section
            and opp_lane == ego_lane
        ):
            return float(self.dfa.get_risk_cost('collision')), 'collision'

        return 0.0, 'drivable'

    # ------------------------------------------------------------------
    # Accumulated horizon risk
    # ------------------------------------------------------------------

    @staticmethod
    def predicted_horizon_risk(stage_costs, gamma: float) -> float:
        """
        Discounted sum of per-cell risk costs over the look-ahead horizon.

            R = Σ_k  γ^k · cost_k

        A larger gamma weighs distant cells more heavily (longer foresight).
        """
        total = 0.0
        discount = 1.0
        for c in stage_costs:
            total += discount * float(c)
            discount *= gamma
        return total

    # ------------------------------------------------------------------
    # Mode selection (purely threshold-based)
    # ------------------------------------------------------------------

    @staticmethod
    def select_mode(
        predicted_risk: float,
        soft_threshold: float,
        hard_threshold: float,
    ) -> RiskLevel:
        """
        Convert accumulated risk to operation mode.

        Higher thresholds → car tolerates more risk → brakes closer.
        Lower thresholds  → car reacts earlier    → brakes farther away.
        """
        if predicted_risk <= soft_threshold:
            return RiskLevel.NOMINAL
        if predicted_risk <= hard_threshold:
            return RiskLevel.CAUTION
        return RiskLevel.BRAKE

    # ------------------------------------------------------------------
    # Lane-change suggestion
    # ------------------------------------------------------------------

    def suggest_lane(
        self,
        ego_lane: int,
        ped_lane: Optional[int],
    ) -> Optional[int]:
        """
        Return a safer adjacent drivable lane, or None.

        Picks the drivable lane that moves away from the pedestrian's lane.
        """
        candidates = []
        for dl in (-1, +1):
            adj = ego_lane + dl
            if 0 <= adj < self.n_lanes and adj in self.drivable_lanes:
                candidates.append(adj)
        if not candidates:
            return None
        if ped_lane is not None:
            candidates.sort(key=lambda l: abs(l - ped_lane), reverse=True)
        return candidates[0]
