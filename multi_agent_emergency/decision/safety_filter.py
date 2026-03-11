"""
safety_filter.py
================
Online safety filter that monitors the ego car's environment and
intervenes when necessary.

The safety filter does NOT do any online optimisation.  It uses a simple
crash-cost hierarchy to decide which lane to steer toward when no
completely safe option exists:

    Crash cost hierarchy (lower = preferred)
    -----------------------------------------
      non-drivable area   :  200
      another car          :  600
      pedestrian           : 1000

If a collision is detected, the simulation should be terminated with
an appropriate error message by the caller.

Intervention levels
-------------------
    NOMINAL   – no intervention, follow DP policy
    CAUTION   – reduce speed, stay in lane
    BRAKE     – emergency stop + suggest lane change
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np


# ── Crash cost constants (lower = less bad) ──────────────────────────
CRASH_COST_NONDRIVABLE  = 200.0
CRASH_COST_CAR          = 600.0
CRASH_COST_PEDESTRIAN   = 1000.0


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
    Online safety filter.

    Parameters
    ----------
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
        graph,
        lane_width: float,
        n_lanes: int,
        drivable_lanes: set,
        warn_distance: float = 12.0,
        brake_distance: float = 6.0,
        caution_speed_factor: float = 0.4,
    ) -> None:
        self.graph = graph
        self.lane_width = lane_width
        self.n_lanes = n_lanes
        self.drivable_lanes = set(drivable_lanes)
        self.warn_distance = warn_distance
        self.brake_distance = brake_distance
        self.caution_speed_factor = caution_speed_factor

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        d_ego: float,
        current_lane: int,
        ped_distance: Optional[float] = None,
        ped_on_lane: Optional[int] = None,
    ) -> Tuple[RiskLevel, dict]:
        """
        Evaluate the current risk level.

        Parameters
        ----------
        d_ego        : lateral deviation from lane centre [m]
        current_lane : current lane index
        ped_distance : distance to pedestrian along driving direction [m],
                       None if no pedestrian nearby
        ped_on_lane  : which lane the pedestrian is on (None if far away)

        Returns
        -------
        level : RiskLevel
        info  : dict with diagnostic details
        """
        info = {'d_ego': d_ego, 'lane': current_lane,
                'ped_dist': ped_distance, 'ped_lane': ped_on_lane}

        # Check if ego is near boundary of its lane
        half_w = self.lane_width / 2.0
        at_boundary = abs(d_ego) > 0.85 * half_w

        # Pedestrian proximity
        ped_ahead = (ped_distance is not None
                     and ped_on_lane == current_lane)

        if ped_ahead and ped_distance < self.brake_distance:
            return RiskLevel.BRAKE, info
        if ped_ahead and ped_distance < self.warn_distance:
            return RiskLevel.CAUTION, info
        if at_boundary and current_lane not in self.drivable_lanes:
            return RiskLevel.CAUTION, info

        return RiskLevel.NOMINAL, info

    # ------------------------------------------------------------------
    # Speed override
    # ------------------------------------------------------------------

    def filter_speed(self, v_s_nominal: float, level: RiskLevel) -> float:
        """Adjust speed based on risk level."""
        if level == RiskLevel.BRAKE:
            return 0.0
        elif level == RiskLevel.CAUTION:
            return v_s_nominal * self.caution_speed_factor
        return v_s_nominal

    # ------------------------------------------------------------------
    # Lane suggestion using crash-cost hierarchy
    # ------------------------------------------------------------------

    def suggest_lane(
        self,
        current_lane: int,
        level: RiskLevel,
        ped_on_lane: Optional[int] = None,
    ) -> int:
        """
        Suggest the safest lane to drive on.

        NOMINAL  -> keep current lane (or nearest drivable if off-road)
        CAUTION / BRAKE -> pick lane with lowest crash cost

        Cost per lane:
          - pedestrian on that lane     → CRASH_COST_PEDESTRIAN
          - lane is non-drivable        → CRASH_COST_NONDRIVABLE
          - else                        → 0
        """
        if level == RiskLevel.NOMINAL:
            if current_lane in self.drivable_lanes:
                return current_lane
            # Snap to nearest drivable
            return min(self.drivable_lanes,
                       key=lambda l: abs(l - current_lane))

        # Evaluate all reachable lanes (current + adjacent)
        candidates = [current_lane] + self.graph.adjacent_lanes(current_lane)
        best_lane = current_lane
        best_cost = float('inf')

        for lane in candidates:
            cost = 0.0
            if lane == ped_on_lane:
                cost += CRASH_COST_PEDESTRIAN
            if lane not in self.drivable_lanes:
                cost += CRASH_COST_NONDRIVABLE
            if cost < best_cost:
                best_cost = cost
                best_lane = lane

        return best_lane

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def check_collision(
        self,
        d_ego: float,
        current_lane: int,
        ped_distance: Optional[float] = None,
        ped_on_lane: Optional[int] = None,
        collision_radius: float = 1.5,
    ) -> Optional[str]:
        """
        Check if a collision has occurred.

        Returns
        -------
        None if no collision, otherwise a string describing the crash:
            "Crashed into pedestrian"
            "Crashed into non-drivable area"
        """
        # Pedestrian collision
        if (ped_distance is not None
            and ped_on_lane == current_lane
            and ped_distance < collision_radius):
            return "Crashed into pedestrian"

        # Off-road: ego beyond lane boundary on non-drivable side
        half_w = self.lane_width / 2.0
        if abs(d_ego) > half_w and current_lane not in self.drivable_lanes:
            return "Crashed into non-drivable area"

        return None

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def summary_line(self, level: RiskLevel, info: dict) -> str:
        """One-line summary for logging."""
        return (
            f"[SafetyFilter] {level.name}  "
            f"lane={info['lane']}  d={info['d_ego']:.2f}  "
            f"ped_dist={info.get('ped_dist', '-')}  "
            f"ped_lane={info.get('ped_lane', '-')}"
        )
