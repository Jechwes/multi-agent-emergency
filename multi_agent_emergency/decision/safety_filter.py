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
    CAUTION   – an obstacle is within warn distance, reduce speed
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
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        d_ego: float,
        current_lane: int,
        ped_distance: Optional[float] = None,
        ped_on_lane: Optional[int] = None,
        ped_target_lane: Optional[int] = None,
        car_distance: Optional[float] = None,
        car_on_lane: Optional[int] = None,
        collision_radius: float = 1.5,
    ) -> Tuple[RiskLevel, dict]:
        """
        Evaluate the current risk level.

        Two separate classifications are produced:

        ``dfa_label``  –  The **actual** DFA letter based on whether an
            atomic proposition is currently violated (collision / off-road).
            This is used by the caller to advance the DFA state.

        ``threat``  –  A **proximity-based** warning describing what the
            closest hazard is.  This determines the RiskLevel and drives
            the speed / lane-change interventions.

        The separation ensures the DFA only transitions to q_fail on a
        real specification violation, while the safety filter can still
        react early via CAUTION / BRAKE.

        Returns
        -------
        level : RiskLevel
        info  : dict with 'dfa_label', 'threat', and diagnostic data
        """
        half_w = self.lane_width / 2.0

        # ---- Actual DFA label (current AP satisfaction) ----
        # Collision only if obstacle is on the SAME lane and within radius.
        # Adjacent-lane obstacles are not collisions (lanes are 4m wide).
        ped_collision = (ped_on_lane is not None
                         and ped_distance is not None
                         and ped_on_lane == current_lane
                         and ped_distance < collision_radius)
        car_collision = (car_on_lane is not None
                         and car_distance is not None
                         and car_on_lane == current_lane
                         and car_distance < collision_radius)

        dfa_label = self.dfa.classify_state(
            d_ego=d_ego,
            lane_half_width=half_w,
            lane_drivable=(current_lane in self.drivable_lanes),
            ped_nearby=ped_collision,
            car_nearby=car_collision,
        )

        # ---- Proximity-based threat assessment ----
        ped_is_threat = False
        if ped_on_lane is not None:
            if ped_on_lane in self.drivable_lanes:
                ped_is_threat = True
            elif ped_target_lane is not None:
                # If pedestrian is outside the drivable lanes but moving towards them
                min_drivable = min(self.drivable_lanes) if self.drivable_lanes else current_lane
                max_drivable = max(self.drivable_lanes) if self.drivable_lanes else current_lane
                
                if ped_on_lane > max_drivable and ped_target_lane < ped_on_lane:
                    ped_is_threat = True
                elif ped_on_lane < min_drivable and ped_target_lane > ped_on_lane:
                    ped_is_threat = True
                else:
                    ped_is_threat = False
            else:
                ped_is_threat = False

        car_on_same_lane = (car_on_lane is not None
                            and car_on_lane == current_lane)

        ped_close = (ped_is_threat and ped_distance is not None
                     and ped_distance < self.warn_distance)
        car_close = (car_on_same_lane and car_distance is not None
                     and car_distance < self.warn_distance)

        if ped_close:
            threat = 'pedestrian'
        elif car_close:
            threat = 'other_car'
        elif current_lane not in self.drivable_lanes:
            threat = 'non_drivable'
        else:
            threat = 'safe'

        risk_cost = self.dfa.get_risk_cost(threat)

        info = {
            'd_ego': d_ego, 'lane': current_lane,
            'ped_dist': ped_distance, 'ped_lane': ped_on_lane,
            'car_dist': car_distance, 'car_lane': car_on_lane,
            'dfa_label': dfa_label, 'threat': threat,
            'risk_cost': risk_cost,
        }

        # --- BRAKE: obstacle within brake distance on SAME lane
        if (ped_is_threat and ped_distance is not None
                and ped_distance < self.brake_distance):
            return RiskLevel.BRAKE, info
        if (car_on_same_lane and car_distance is not None
                and car_distance < self.brake_distance):
            return RiskLevel.BRAKE, info

        # --- CAUTION: obstacle within warn distance or non-drivable
        if threat != 'safe':
            return RiskLevel.CAUTION, info

        return RiskLevel.NOMINAL, info

    # ------------------------------------------------------------------
    # Policy Selection (Supervisory Control)
    # ------------------------------------------------------------------

    def choose_policy(self, ped_distance: float = None, lookahead_distance: float = 60.0) -> str:
        """
        Acts as a Supervisory Controller.
        Chooses which DP policy to execute based on sensor lookahead.
        """
        if ped_distance is not None and ped_distance <= lookahead_distance:
            return 'evasive'
        return 'nominal'

    # ------------------------------------------------------------------
    # Lane suggestion using DFA risk costs
    # ------------------------------------------------------------------

    def suggest_lane(
        self,
        current_lane: int,
        level: RiskLevel,
        ped_on_lane: Optional[int] = None,
        ped_distance: Optional[float] = None,
        ped_target_lane: Optional[int] = None,
        car_on_lane: Optional[int] = None,
        car_distance: Optional[float] = None,
    ) -> int:
        """
        Suggest the safest lane to drive on.

        Decision logic with distance zones:

          NOMINAL → stay in current lane.

          CAUTION with obstacle far (> lane_change_distance)
            → if a clear drivable lane exists, suggest it

          CAUTION with obstacle close (≤ lane_change_distance),
          or BRAKE
            → too late to change lane; stay put and brake.

        A lane is blocked if the pedestrian is on it **or** the
        pedestrian is moving toward it (``ped_target_lane``).
        If the pedestrian is moving away, the current lane is free.

        Non-drivable lanes are **never** suggested.
        """
        # NOMINAL → keep current lane
        if level == RiskLevel.NOMINAL:
            if current_lane in self.drivable_lanes:
                return current_lane
            return min(self.drivable_lanes,
                       key=lambda l: abs(l - current_lane))

        # BRAKE → always stay straight, never start a lane change
        if level == RiskLevel.BRAKE:
            return current_lane

        # CAUTION → only suggest lane change if far enough away
        # Determine the nearest threat distance
        threat_dist = float('inf')
        
        # Always stay in lane for pedestrians (brake instead of swerving).
        # Swerving is only considered for car threats.
        if car_distance is not None:
            threat_dist = min(threat_dist, car_distance)

        if threat_dist == float('inf'):
            # Threat is pedestrian or non-drivable lane.
            # If off-road, recover gracefully. Otherwise stay put.
            if current_lane not in self.drivable_lanes:
                return min(self.drivable_lanes, key=lambda l: abs(l - current_lane))
            return current_lane

        if threat_dist < self.lane_change_distance:
            # Too close — stay straight, brake will handle it
            return current_lane

        # Far enough: look for a clear drivable alternative
        adjacent = self.graph.adjacent_lanes(current_lane)

        def _is_blocked(lane: int) -> bool:
            # Blocked if ped is currently on it
            if lane == ped_on_lane:
                return True
            # Blocked if ped is heading toward it
            if ped_target_lane is not None and lane == ped_target_lane:
                return True
            if lane == car_on_lane:
                return True
            return False

        safe_alternatives = [
            l for l in adjacent
            if l in self.drivable_lanes and not _is_blocked(l)
        ]

        if safe_alternatives:
            return min(safe_alternatives,
                       key=lambda l: abs(l - current_lane))

        # No clear lane → stay put
        return current_lane

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

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def summary_line(self, level: RiskLevel, info: dict) -> str:
        """One-line summary for logging."""
        return (
            f"[SafetyFilter] {level.name}  "
            f"threat={info.get('threat', '?')}  "
            f"dfa={info.get('dfa_label', '?')}  "
            f"lane={info['lane']}  d={info['d_ego']:.2f}  "
            f"ped_dist={info.get('ped_dist', '-')}  "
            f"ped_lane={info.get('ped_lane', '-')}  "
            f"car_dist={info.get('car_dist', '-')}  "
            f"car_lane={info.get('car_lane', '-')}"
        )
