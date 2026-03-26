"""
safety_filter.py
================
Online safety filter that monitors the ego car's environment and
intervenes when necessary.

The safety filter is **connected to the co-safety DFA**:

    φ = G( ¬non_drivable  ∧  ¬pedestrian  ∧  ¬other_car )

At every time step, the filter calls ``dfa.classify_state()`` to
determine the current label.  The associated ``dfa.risk_cost`` for that
label is used for:
  1. grading the *severity* of the current situation
  2. ranking lane-change candidates by expected violation cost

The DFA state ``q_current`` is advanced by the decision maker; the
safety filter only reads it to detect if the spec has already been
violated (q == sink).

Intervention levels
-------------------
    NOMINAL   – label is 'safe', no intervention
    CAUTION   – predicted risk is elevated, use evasive policy
    BRAKE     – an obstacle is within brake distance, emergency stop

The filter also suggests the safest reachable lane by summing the
DFA risk costs of each candidate lane's expected label.
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


# ── Collision error ──────────────────────────────────────────────────
class CrashError(Exception):
    """Raised when a collision is detected.  Contains a message
    describing what the ego car crashed into."""
    pass


# ── SafetyFilter ─────────────────────────────────────────────────────
class SafetyFilter:
    """
    Online safety filter connected to the co-safety DFA.

    Parameters
    ----------
    dfa              : RoundaboutDFA (provides classify_state + risk_cost)
    graph            : LaneletGraph (lane connectivity)
    lane_width       : lane width [m]
    n_lanes          : total number of lanes
    drivable_lanes   : set of normally-drivable lane indices
    warn_distance    : distance to obstacle triggering CAUTION [m]
    brake_distance   : distance to obstacle triggering BRAKE [m]
    caution_speed_factor : MPC speed multiplied by this during CAUTION
    """

    def __init__(
        self,
        dfa,
        graph,
        lane_width: float,
        n_lanes: int,
        drivable_lanes: set,
        warn_distance: float = 12.0,
        brake_distance: float = 3.0,
        lane_change_distance: float = 8.0,
        caution_speed_factor: float = 0.4,
    ) -> None:
        self.dfa = dfa
        self.graph = graph
        self.lane_width = lane_width
        self.n_lanes = n_lanes
        self.drivable_lanes = set(drivable_lanes)
        self.warn_distance = warn_distance
        self.brake_distance = brake_distance
        self.lane_change_distance = lane_change_distance
        self.caution_speed_factor = caution_speed_factor

    # ------------------------------------------------------------------
    # MPC-predictive risk scoring
    # ------------------------------------------------------------------

    def stage_risk_cost(
        self,
        d_ego: float,
        current_lane: int,
        ped_distance: Optional[float] = None,
        ped_on_lane: Optional[int] = None,
        ped_target_lane: Optional[int] = None,
        car_distance: Optional[float] = None,
        car_on_lane: Optional[int] = None,
        collision_radius: float = 1.5,
    ) -> Tuple[float, str]:
        """
        Compute a per-step risk cost for MPC horizon evaluation.

        The returned cost uses DFA risk costs for hard violations and a
        distance-weighted fraction of the same costs for near-miss cases.
        """
        half_w = self.lane_width / 2.0

        ped_collision = (
            ped_on_lane is not None
            and ped_distance is not None
            and ped_on_lane == current_lane
            and ped_distance < collision_radius
        )
        car_collision = (
            car_on_lane is not None
            and car_distance is not None
            and car_on_lane == current_lane
            and car_distance < collision_radius
        )

        dfa_label = self.dfa.classify_state(
            d_ego=d_ego,
            lane_half_width=half_w,
            lane_drivable=(current_lane in self.drivable_lanes),
            ped_nearby=ped_collision,
            car_nearby=car_collision,
        )

        if dfa_label != 'safe':
            return float(self.dfa.get_risk_cost(dfa_label)), dfa_label

        # Score proximity cost for any pedestrian on a drivable lane OR
        # moving toward one, not just same-lane pedestrians.
        ped_is_threat = False
        if ped_on_lane is not None:
            if ped_on_lane in self.drivable_lanes:
                ped_is_threat = True
            elif ped_target_lane is not None:
                min_drivable = min(self.drivable_lanes) if self.drivable_lanes else current_lane
                max_drivable = max(self.drivable_lanes) if self.drivable_lanes else current_lane
                if ped_on_lane > max_drivable and ped_target_lane < ped_on_lane:
                    ped_is_threat = True
                elif ped_on_lane < min_drivable and ped_target_lane > ped_on_lane:
                    ped_is_threat = True

        ped_prox_cost = 0.0
        if (
            ped_is_threat
            and ped_distance is not None
            and ped_distance < self.warn_distance
        ):
            span = max(self.warn_distance - self.brake_distance, 1e-6)
            alpha = (self.warn_distance - ped_distance) / span
            alpha = float(np.clip(alpha, 0.0, 1.0))
            ped_prox_cost = alpha * float(self.dfa.get_risk_cost('pedestrian'))

        car_prox_cost = 0.0
        if (
            car_on_lane is not None
            and car_on_lane == current_lane
            and car_distance is not None
            and car_distance < self.warn_distance
        ):
            span = max(self.warn_distance - self.brake_distance, 1e-6)
            alpha = (self.warn_distance - car_distance) / span
            alpha = float(np.clip(alpha, 0.0, 1.0))
            car_prox_cost = alpha * float(self.dfa.get_risk_cost('other_car'))

        if ped_prox_cost >= car_prox_cost and ped_prox_cost > 0.0:
            return ped_prox_cost, 'pedestrian'
        if car_prox_cost > 0.0:
            return car_prox_cost, 'other_car'
        return 0.0, 'safe'

    @staticmethod
    def predicted_horizon_risk(stage_costs, gamma: float) -> float:
        """Discounted predictive risk over an MPC horizon."""
        total = 0.0
        discount = 1.0
        for c in stage_costs:
            total += discount * float(c)
            discount *= float(gamma)
        return float(total)

    @staticmethod
    def select_mode(
        predicted_risk: float,
        soft_threshold: float,
        hard_threshold: float,
    ) -> RiskLevel:
        """
        Convert predicted risk to operation mode.

        NOMINAL: predicted_risk <= soft_threshold
        CAUTION: soft_threshold < predicted_risk <= hard_threshold
        BRAKE:   predicted_risk > hard_threshold
        """
        if predicted_risk <= soft_threshold:
            return RiskLevel.NOMINAL
        if predicted_risk <= hard_threshold:
            return RiskLevel.CAUTION
        return RiskLevel.BRAKE

    # ------------------------------------------------------------------
    # Collision detection (uses DFA classification)
    # ------------------------------------------------------------------

    def check_collision(
        self,
        d_ego: float,
        current_lane: int,
        ped_distance: Optional[float] = None,
        ped_on_lane: Optional[int] = None,
        car_distance: Optional[float] = None,
        car_on_lane: Optional[int] = None,
        collision_radius: float = 1.5,
    ) -> Optional[str]:
        """
        Check if a collision has occurred.

        Returns None if no collision, otherwise a string matching the
        DFA letter that was violated.
        """
        half_w = self.lane_width / 2.0

        # Pedestrian collision — same lane only
        if (ped_distance is not None
            and ped_on_lane is not None
            and ped_on_lane == current_lane
            and ped_distance < collision_radius):
            return "Crashed into pedestrian (label: 'pedestrian')"

        # Other car collision — same lane only
        if (car_distance is not None
            and car_on_lane is not None
            and car_on_lane == current_lane
            and car_distance < collision_radius):
            return "Crashed into other car (label: 'other_car')"

        # Off-road
        if abs(d_ego) > half_w and current_lane not in self.drivable_lanes:
            return "Crashed into non-drivable area (label: 'non_drivable')"

        return None
