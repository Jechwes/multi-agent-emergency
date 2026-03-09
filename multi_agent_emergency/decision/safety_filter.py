"""
safety_filter.py
================
Online safety filter that monitors the risk value function produced by
the ``RoundaboutDPDecisionMaker`` and can override the nominal driving
action when the situation becomes too risky.

Architecture
------------
Every simulation tick the filter is called with the current Frenet state
(s, d) and the pedestrian lateral position (p_lateral).  It queries
``maker.get_risk_at(s, d, p)`` and compares against three thresholds:

    risk ≤ WARN_THRESH          → NOMINAL   (no intervention)
    WARN_THRESH < risk ≤ BRAKE  → CAUTION   (reduce speed)
    risk > BRAKE_THRESH         → BRAKE     (full stop + lane shift)

When intervening the filter can:
  • Reduce the MPC speed reference  (caution)
  • Force a full brake              (emergency)
  • Shift the lateral reference     (evasive lane change)
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np


# ── Risk level enum ──────────────────────────────────────────────────
class RiskLevel(Enum):
    NOMINAL  = auto()   # safe – no intervention
    CAUTION  = auto()   # elevated risk – slow down
    BRAKE    = auto()   # high risk – full stop


# ── SafetyFilter ─────────────────────────────────────────────────────
class SafetyFilter:
    """
    Online safety filter that monitors the DP risk value function.

    Parameters
    ----------
    maker        : RoundaboutDPDecisionMaker (must have compute_risk_field done)
    warn_thresh  : combined risk above which → CAUTION (speed reduction)
    brake_thresh : combined risk above which → BRAKE   (emergency stop)
    caution_speed_factor : MPC speed is multiplied by this during CAUTION
    """

    def __init__(
        self,
        maker,
        warn_thresh:  float = 15.0,
        brake_thresh: float = 40.0,
        caution_speed_factor: float = 0.4,
    ) -> None:
        self.maker = maker
        self.warn_thresh  = warn_thresh
        self.brake_thresh = brake_thresh
        self.caution_speed_factor = caution_speed_factor

        # Track previous risk for logging trend
        self._prev_risk: float = 0.0

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        s: float,
        d: float,
        p_lateral: Optional[float] = None,
    ) -> Tuple[RiskLevel, float, dict]:
        """
        Evaluate risk at the current state.

        Parameters
        ----------
        s          : longitudinal Frenet coordinate
        d          : lateral Frenet coordinate
        p_lateral  : pedestrian lateral position (in same lateral-axis
                     frame as d), or None if no pedestrian

        Returns
        -------
        level      : RiskLevel enum
        risk_total : combined scalar risk
        components : per-dimension breakdown dict
        """
        components = self.maker.get_risk_components(s, d, p_lateral)
        risk_total = components['total']
        self._prev_risk = risk_total

        if risk_total > self.brake_thresh:
            level = RiskLevel.BRAKE
        elif risk_total > self.warn_thresh:
            level = RiskLevel.CAUTION
        else:
            level = RiskLevel.NOMINAL

        return level, risk_total, components

    # ------------------------------------------------------------------
    # Action override
    # ------------------------------------------------------------------

    def filter_speed(
        self,
        v_s_nominal: float,
        level: RiskLevel,
    ) -> float:
        """
        Adjust the longitudinal speed reference based on risk level.

        NOMINAL  → pass through
        CAUTION  → reduce speed by caution factor
        BRAKE    → 0 m/s  (full stop)
        """
        if level == RiskLevel.BRAKE:
            return 0.0
        elif level == RiskLevel.CAUTION:
            return v_s_nominal * self.caution_speed_factor
        else:
            return v_s_nominal

    def suggest_d_ref(
        self,
        d_ref_nominal: float,
        level: RiskLevel,
        p_lateral: Optional[float] = None,
    ) -> float:
        """
        Suggest a lateral reference that steers away from danger.

        NOMINAL  → keep nominal d_ref
        CAUTION  → commit to safest lane (decisive lane change)
        BRAKE    → commit to safest lane
        """
        if level == RiskLevel.NOMINAL:
            return d_ref_nominal

        d_safe = self.maker.get_safest_lane_d(p_lateral)
        return d_safe

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def summary_line(
        self,
        level: RiskLevel,
        risk_total: float,
        components: dict,
    ) -> str:
        """One-line risk summary for logging."""
        tag = level.name
        return (
            f"[SafetyFilter] {tag}  "
            f"risk={risk_total:.1f}  "
            f"(s={components['s']:.1f}  d={components['d']:.1f}  "
            f"p={components['p']:.1f})  "
            f"thresholds: warn={self.warn_thresh}, brake={self.brake_thresh}"
        )
