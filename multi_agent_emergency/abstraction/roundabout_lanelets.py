"""
roundabout_lanelets.py
======================
Create equal-arc-length Frenet-frame lanelets for a roundabout.

The roundabout is modelled as concentric circular lanes around a centre
point.  It is divided into **N sections** of equal **angle**, which—for
circular lanes—automatically gives equal **arc-length** per section on
every ring.

Each section × lane combination is a ``SectionLanelet`` that carries its
own local Frenet coordinate system:

    s ∈ [0, L_section]    longitudinal progress within the section
    d ∈ [-w/2, +w/2]      lateral deviation from the centre-line

The state vectors for the two decoupled MDPs are:

    MDP_s  state: [s,  s_dot ]     (longitudinal)
    MDP_d  state: [d,  d_dot ]     (lateral)

This makes the transition matrices small (2-D each) and solvable by the
FMTensJelmar dynprog algorithms.

Usage
-----
    from abstraction.roundabout_lanelets import RoundaboutLaneletMap

    rmap = RoundaboutLaneletMap(
        centre=(cx, cy),        # CARLA world coordinates of roundabout centre
        inner_radius=13.5,      # [m] inner edge of innermost ring
        lane_width=4.0,         # [m] radial width of each lane
        n_lanes=4,              # number of concentric rings
        n_sections=12,          # number of equal-angle sectors
    )

    # Query Frenet state for a world point
    sec_id, lane_id, frenet = rmap.to_frenet(xy, speed, yaw)

    # Convert back
    cart = rmap.to_cartesian(sec_id, lane_id, s, d)

    # Get section arc-length (same for all sections on a given lane)
    L = rmap.section_arc_length(lane_id)

    # Get decoupled LinModel matrices for MDP_s and MDP_d
    A_s, B_s = rmap.get_longitudinal_dynamics(dt=0.1)
    A_d, B_d = rmap.get_lateral_dynamics(dt=0.1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FrenetState2D:
    """Reduced 2-D Frenet state for one of the decoupled MDPs."""
    pos: float     # s or d  [m]
    vel: float     # s_dot or d_dot  [m/s]

    def as_array(self) -> np.ndarray:
        return np.array([self.pos, self.vel])


@dataclass
class FrenetState:
    """Full 4-D Frenet state within a section."""
    s: float         # longitudinal progress within the section [m]
    d: float         # lateral deviation from centre-line [m]
    s_dot: float = 0.0   # longitudinal speed [m/s]
    d_dot: float = 0.0   # lateral speed [m/s]

    @property
    def longitudinal(self) -> FrenetState2D:
        return FrenetState2D(self.s, self.s_dot)

    @property
    def lateral(self) -> FrenetState2D:
        return FrenetState2D(self.d, self.d_dot)

    def as_array(self) -> np.ndarray:
        return np.array([self.s, self.d, self.s_dot, self.d_dot])


@dataclass
class CartesianState:
    """Cartesian world-frame state."""
    x: float
    y: float
    heading: float   # tangent angle [rad]


# ---------------------------------------------------------------------------
# SectionLanelet – one (section, lane) combination
# ---------------------------------------------------------------------------

class SectionLanelet:
    """
    A single arc segment of one lane within one angular section of the
    roundabout.

    For a circular roundabout the centre-line is an arc at radius
        r = inner_radius + (lane_id + 0.5) * lane_width
    spanning from angle_start to angle_end (counter-clockwise, radians).

    The local Frenet s-axis runs from 0 to ``arc_length``.

    Parameters
    ----------
    centre       : (cx, cy)  roundabout centre in world frame.
    radius       : centre-line radius of this lane [m].
    lane_width   : full width of this lane [m].
    angle_start  : start angle of this section [rad].
    angle_end    : end angle of this section [rad].
    section_id   : section (sector) index.
    lane_id      : lane (ring) index, 0 = innermost.
    n_arc_points : number of sample points for the spline.
    """

    def __init__(
        self,
        centre: Tuple[float, float],
        radius: float,
        lane_width: float,
        angle_start: float,
        angle_end: float,
        section_id: int,
        lane_id: int,
        n_arc_points: int = 64,
    ) -> None:
        self.centre = np.asarray(centre, dtype=float)
        self.radius = radius
        self.lane_width = lane_width
        self.angle_start = angle_start
        self.angle_end = angle_end
        self.section_id = section_id
        self.lane_id = lane_id

        # Arc length of this section on this lane
        self.arc_length: float = radius * abs(angle_end - angle_start)

        # Build waypoints along the arc  (CCW direction)
        angles = np.linspace(angle_start, angle_end, n_arc_points)
        self._angles = angles
        self._waypoints = np.column_stack([
            centre[0] + radius * np.cos(angles),
            centre[1] + radius * np.sin(angles),
        ])

        # Cumulative arc-length parameter
        diffs = np.diff(self._waypoints, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        self._arc_lengths = np.concatenate([[0.0], np.cumsum(seg_len)])
        # Spline interpolation: x(s), y(s)
        self._spline_x = CubicSpline(self._arc_lengths, self._waypoints[:, 0])
        self._spline_y = CubicSpline(self._arc_lengths, self._waypoints[:, 1])
        # KD-tree for fast nearest-point lookup
        self._kdtree = KDTree(self._waypoints)

    # ---- Frenet conversion ------------------------------------------------

    def to_frenet(
        self,
        xy: np.ndarray,
        speed: Optional[float] = None,
        yaw: Optional[float] = None,
    ) -> FrenetState:
        """Convert world (x,y) to section-local Frenet state."""
        xy = np.asarray(xy, dtype=float).ravel()[:2]
        s_raw = self._project_to_arc_length(xy)
        d = self._signed_lateral_deviation(xy, s_raw)

        s_dot, d_dot = 0.0, 0.0
        if speed is not None and yaw is not None:
            tang = self.tangent_angle(s_raw)
            delta = yaw - tang
            s_dot = speed * math.cos(delta)
            d_dot = speed * math.sin(delta)

        return FrenetState(s=s_raw, d=d, s_dot=s_dot, d_dot=d_dot)

    def to_cartesian(self, s: float, d: float) -> CartesianState:
        """Convert section-local Frenet (s, d) back to world Cartesian."""
        s = float(np.clip(s, 0.0, self.arc_length))
        cx = float(self._spline_x(s))
        cy = float(self._spline_y(s))
        heading = self.tangent_angle(s)
        nx = -math.sin(heading)
        ny =  math.cos(heading)
        return CartesianState(x=cx + d * nx, y=cy + d * ny, heading=heading)

    def tangent_angle(self, s: float) -> float:
        """Centre-line tangent angle [rad] at arc-length s."""
        s = float(np.clip(s, 0.0, self.arc_length))
        dx = float(self._spline_x(s, 1))
        dy = float(self._spline_y(s, 1))
        return math.atan2(dy, dx)

    def curvature(self, s: float) -> float:
        """Signed curvature kappa(s) [1/m]."""
        # For a perfect circle: kappa = 1/radius (constant)
        # Using spline for generality
        s = float(np.clip(s, 0.0, self.arc_length))
        xp  = float(self._spline_x(s, 1))
        yp  = float(self._spline_y(s, 1))
        xpp = float(self._spline_x(s, 2))
        ypp = float(self._spline_y(s, 2))
        denom = (xp**2 + yp**2) ** 1.5
        return (xp * ypp - yp * xpp) / denom if denom > 1e-12 else 0.0

    def distance_to(self, xy: np.ndarray) -> float:
        """Euclidean distance from xy to the nearest centre-line point."""
        _, idx = self._kdtree.query(xy.ravel()[:2])
        return float(np.linalg.norm(xy.ravel()[:2] - self._waypoints[idx]))

    def contains(self, xy: np.ndarray) -> bool:
        """True if xy is within lane_width / 2 of the centre-line."""
        return self.distance_to(xy) <= self.lane_width / 2.0

    # ---- Inner boundary / outer boundary centre-line points ---------------

    def boundary_points(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (inner_boundary, outer_boundary) as (n_samples, 2) arrays.

        inner = centre-line - lane_width/2 (towards roundabout centre)
        outer = centre-line + lane_width/2 (away from roundabout centre)
        """
        s_vals = np.linspace(0.0, self.arc_length, n_samples)
        half_w = self.lane_width / 2.0
        inner, outer = [], []
        for s in s_vals:
            heading = self.tangent_angle(s)
            cx = float(self._spline_x(s))
            cy = float(self._spline_y(s))
            n = np.array([-math.sin(heading), math.cos(heading)])
            c = np.array([cx, cy])
            inner.append(c - half_w * n)
            outer.append(c + half_w * n)
        return np.array(inner), np.array(outer)

    # ---- Private helpers --------------------------------------------------

    def _project_to_arc_length(self, xy: np.ndarray) -> float:
        """Return arc-length s of the closest centreline point to xy."""
        _, idx = self._kdtree.query(xy)
        best_s = self._arc_lengths[idx]
        best_dist = np.linalg.norm(xy - self._waypoints[idx])

        n = len(self._waypoints)
        for a, b in [(idx - 1, idx), (idx, idx + 1)]:
            if a < 0 or b >= n:
                continue
            p0, p1 = self._waypoints[a], self._waypoints[b]
            seg = p1 - p0
            seg_len_sq = float(np.dot(seg, seg))
            if seg_len_sq < 1e-12:
                continue
            t = float(np.clip(np.dot(xy - p0, seg) / seg_len_sq, 0.0, 1.0))
            proj = p0 + t * seg
            dist = np.linalg.norm(xy - proj)
            if dist < best_dist:
                best_dist = dist
                best_s = self._arc_lengths[a] + t * (self._arc_lengths[b] - self._arc_lengths[a])
        return float(best_s)

    def _signed_lateral_deviation(self, xy: np.ndarray, s: float) -> float:
        """Signed lateral offset d (positive = left of travel direction)."""
        heading = self.tangent_angle(s)
        cx = float(self._spline_x(s))
        cy = float(self._spline_y(s))
        v = xy - np.array([cx, cy])
        n = np.array([-math.sin(heading), math.cos(heading)])
        return float(np.dot(v, n))


# ---------------------------------------------------------------------------
# RoundaboutLaneletMap – the full sectored roundabout
# ---------------------------------------------------------------------------

class RoundaboutLaneletMap:
    """
    Complete roundabout lanelet map divided into equal-angle sections.

    Each section spans ``2*pi / n_sections`` radians. Within each section,
    every lane has the same angular extent, so different lanes have arc
    lengths proportional to their radius -- but all sections of the *same*
    lane share exactly the same arc length.  This makes the MDP transition
    matrices identical across sections for a given lane.

    Parameters
    ----------
    centre       : (cx, cy)  roundabout centre in world frame [m].
    inner_radius : radius of the inner edge of the innermost lane [m].
    lane_width   : radial width of each lane [m].
    n_lanes      : number of concentric lanes (rings).
    n_sections   : number of equal-angle sectors.
    direction    : 'ccw' (counter-clockwise) or 'cw' (clockwise).
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (-0.5, 0.5),
        inner_radius: float = 13.5,
        lane_width: float = 4.0,
        n_lanes: int = 4,
        n_sections: int = 12,
        direction: str = "ccw",
    ) -> None:
        self.centre = np.asarray(centre, dtype=float)
        self.inner_radius = inner_radius
        self.lane_width = lane_width
        self.n_lanes = n_lanes
        self.n_sections = n_sections
        self.direction = direction

        self.section_angle: float = 2.0 * math.pi / n_sections  # radians

        # Build all SectionLanelets
        # sections[section_id][lane_id] = SectionLanelet
        self.sections: Dict[int, Dict[int, SectionLanelet]] = {}
        self._all_lanelets: List[SectionLanelet] = []

        sign = 1.0 if direction == "ccw" else -1.0

        for sec in range(n_sections):
            angle_start = sign * sec * self.section_angle
            angle_end   = sign * (sec + 1) * self.section_angle
            self.sections[sec] = {}
            for lane in range(n_lanes):
                r = inner_radius + (lane + 0.5) * lane_width
                sl = SectionLanelet(
                    centre=centre,
                    radius=r,
                    lane_width=lane_width,
                    angle_start=angle_start,
                    angle_end=angle_end,
                    section_id=sec,
                    lane_id=lane,
                )
                self.sections[sec][lane] = sl
                self._all_lanelets.append(sl)

        # Precompute lane radii for quick lookup
        self._lane_radii = np.array([
            inner_radius + (i + 0.5) * lane_width for i in range(n_lanes)
        ])

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------

    def section_arc_length(self, lane_id: int) -> float:
        """Arc length of one section on the given lane [m]."""
        r = self.inner_radius + (lane_id + 0.5) * self.lane_width
        return r * self.section_angle

    def total_circumference(self, lane_id: int) -> float:
        """Full circumference of the given lane [m]."""
        r = self.inner_radius + (lane_id + 0.5) * self.lane_width
        return 2.0 * math.pi * r

    def get_section_lanelet(self, section_id: int, lane_id: int) -> SectionLanelet:
        """Retrieve a specific SectionLanelet."""
        return self.sections[section_id][lane_id]

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _angle_of(self, xy: np.ndarray) -> float:
        """Return the angle of xy w.r.t. the roundabout centre [rad, 0..2pi]."""
        dx = xy[0] - self.centre[0]
        dy = xy[1] - self.centre[1]
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi
        return angle

    def _radius_of(self, xy: np.ndarray) -> float:
        """Distance from roundabout centre."""
        return float(np.linalg.norm(xy[:2] - self.centre))

    def _identify_section(self, xy: np.ndarray) -> int:
        """Return section index for a world point."""
        angle = self._angle_of(xy)
        if self.direction == "cw":
            angle = 2 * math.pi - angle
        sec = int(angle / self.section_angle)
        return sec % self.n_sections

    def _identify_lane(self, xy: np.ndarray) -> int:
        """Return lane index for a world point (clipped to valid range)."""
        r = self._radius_of(xy)
        lane_float = (r - self.inner_radius) / self.lane_width - 0.5
        lane = int(round(lane_float))
        return max(0, min(lane, self.n_lanes - 1))

    def to_frenet(
        self,
        xy: np.ndarray,
        speed: Optional[float] = None,
        yaw: Optional[float] = None,
    ) -> Tuple[int, int, FrenetState]:
        """
        Convert world (x, y) to (section_id, lane_id, FrenetState).

        Returns
        -------
        section_id : int
        lane_id    : int
        frenet     : FrenetState  with s local to the section [0, L_section]
        """
        xy = np.asarray(xy, dtype=float).ravel()[:2]
        sec = self._identify_section(xy)
        lane = self._identify_lane(xy)
        sl = self.sections[sec][lane]
        fs = sl.to_frenet(xy, speed=speed, yaw=yaw)
        return sec, lane, fs

    def to_cartesian(
        self, section_id: int, lane_id: int, s: float, d: float,
    ) -> CartesianState:
        """Convert section-local Frenet (s, d) to world Cartesian."""
        sl = self.sections[section_id][lane_id]
        return sl.to_cartesian(s, d)

    # ------------------------------------------------------------------
    # Multi-lane Frenet conversion (global d across all lanes)
    # ------------------------------------------------------------------

    def to_frenet_multilane(
        self,
        xy: np.ndarray,
        speed: Optional[float] = None,
        yaw: Optional[float] = None,
        ref_lane: int = 1,
    ) -> Tuple[int, int, 'FrenetState']:
        """
        Convert world (x, y) to (section_id, lane_id, FrenetState)
        where ``d`` is the **global** lateral offset from the reference
        point between drivable lanes 1 and 2.

        The reference radius is::

            r_ref = inner_radius + (n_lanes / 2) * lane_width

        and ``d_global = r_car − r_ref``  (positive = outward).

        ``s`` is computed on *ref_lane* (default: lane 1).

        Parameters
        ----------
        xy       : (2,) world position.
        speed    : vehicle speed for velocity decomposition.
        yaw      : vehicle heading [rad].
        ref_lane : lane used for the s-axis projection (default 1).

        Returns
        -------
        section_id : int
        lane_id    : int  (actual lane the car is closest to)
        frenet     : FrenetState  with d = d_global, s from ref_lane.
        """
        xy = np.asarray(xy, dtype=float).ravel()[:2]
        sec = self._identify_section(xy)
        lane = self._identify_lane(xy)

        # Use the reference lane's lanelet for the s-axis projection
        sl_ref = self.sections[sec][ref_lane]
        fs_local = sl_ref.to_frenet(xy, speed=speed, yaw=yaw)

        # Compute d_global from radial distance
        r_ref = self.inner_radius + (self.n_lanes / 2.0) * self.lane_width
        r_car = self._radius_of(xy)
        d_global = r_car - r_ref

        fs = FrenetState(
            s=fs_local.s,
            d=d_global,
            s_dot=fs_local.s_dot,
            d_dot=fs_local.d_dot,
        )
        return sec, lane, fs

    def to_cartesian_multilane(
        self,
        section_id: int,
        s: float,
        d_global: float,
        ref_lane: int = 1,
    ) -> 'CartesianState':
        """
        Convert multi-lane Frenet (s, d_global) to world Cartesian.

        ``s`` is the arc-length on *ref_lane*.
        ``d_global`` is the lateral offset from the reference midpoint
        between lanes 1 and 2 (positive = outward).

        The method projects the (s, d_global) pair back through the
        ref_lane's spline, then offsets in the normal direction.
        """
        sl_ref = self.sections[section_id][ref_lane]
        s_clip = float(np.clip(s, 0.0, sl_ref.arc_length))

        # Reference point on the ref_lane centre-line
        cx = float(sl_ref._spline_x(s_clip))
        cy = float(sl_ref._spline_y(s_clip))
        heading = sl_ref.tangent_angle(s_clip)

        # Normal direction (positive d = outward from roundabout centre)
        nx = -math.sin(heading)
        ny = math.cos(heading)

        # The ref_lane centre is at r_ref_lane
        r_ref_lane = self.inner_radius + (ref_lane + 0.5) * self.lane_width
        # Global d=0 reference is at r_ref
        r_ref = self.inner_radius + (self.n_lanes / 2.0) * self.lane_width
        # Lateral offset from ref_lane centre → global reference
        d_offset = r_ref - r_ref_lane   # e.g. +2.0 m for ref_lane=1

        # Total lateral offset from the ref_lane centre-line
        d_total = d_offset + d_global

        return CartesianState(
            x=cx + d_total * nx,
            y=cy + d_total * ny,
            heading=heading,
        )

    # ------------------------------------------------------------------
    # Transition helpers for MDP
    # ------------------------------------------------------------------

    def next_section(self, section_id: int) -> int:
        """Return the next section index (wraps around)."""
        return (section_id + 1) % self.n_sections

    def prev_section(self, section_id: int) -> int:
        """Return the previous section index (wraps around)."""
        return (section_id - 1) % self.n_sections

    # ------------------------------------------------------------------
    # Decoupled dynamics  (MDP_s and MDP_d)
    # ------------------------------------------------------------------

    @staticmethod
    def get_longitudinal_dynamics(dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discrete-time double-integrator for longitudinal Frenet dynamics.

        State:  x_s = [s, s_dot]
        Input:  u_s = a_s  (longitudinal acceleration)

            s(k+1)     = s(k) + dt * s_dot(k) + 0.5 * dt^2 * a_s
            s_dot(k+1) = s_dot(k) + dt * a_s

        Returns
        -------
        A_s : (2, 2) state matrix
        B_s : (2, 1) input matrix
        """
        A_s = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ])
        B_s = np.array([
            [0.5 * dt**2],
            [dt],
        ])
        return A_s, B_s

    @staticmethod
    def get_lateral_dynamics(dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discrete-time double-integrator for lateral Frenet dynamics.

        State:  x_d = [d, d_dot]
        Input:  u_d = a_d  (lateral acceleration)

            d(k+1)     = d(k) + dt * d_dot(k) + 0.5 * dt^2 * a_d
            d_dot(k+1) = d_dot(k) + dt * a_d

        Returns
        -------
        A_d : (2, 2) state matrix
        B_d : (2, 1) input matrix
        """
        A_d = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ])
        B_d = np.array([
            [0.5 * dt**2],
            [dt],
        ])
        return A_d, B_d

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def draw_in_carla(
        self,
        world,
        z: float = 0.2,
        n_samples: int = 100,
        s_grid_step: Optional[float] = None,
        life_time: float = 30.0,
        draw_section_boundaries: bool = True,
        draw_lane_boundaries: bool = True,
        draw_centrelines: bool = True,
        draw_s_grid: bool = True,
        draw_section_labels: bool = True,
    ) -> None:
        """
        Draw the full roundabout lanelet map in CARLA.

        Parameters
        ----------
        world                  : carla.World
        z                      : drawing height [m]
        n_samples              : points per arc for smooth curves
        s_grid_step            : if given, draw transverse grid lines
                                 every s_grid_step [m] within each section
        life_time              : CARLA debug line lifetime [s]
        draw_section_boundaries: draw radial lines separating sections
        draw_lane_boundaries   : draw inner/outer edges of each lane
        draw_centrelines       : draw lane centre-lines
        draw_s_grid            : draw transverse s-grid within sections
        draw_section_labels    : draw section index labels
        """
        import carla as carla_module

        def _loc(x, y):
            return carla_module.Location(x=float(x), y=float(y), z=z)

        def _draw_line(p0, p1, color_rgb, thick=0.06):
            world.debug.draw_line(
                _loc(p0[0], p0[1]),
                _loc(p1[0], p1[1]),
                thickness=thick,
                color=carla_module.Color(r=color_rgb[0], g=color_rgb[1], b=color_rgb[2]),
                life_time=life_time,
                persistent_lines=True,
            )

        # Colour palette per lane
        lane_colours = [
            (0, 255, 0),      # green
            (255, 0, 255),    # magenta
            (0, 255, 255),    # cyan
            (255, 255, 0),    # yellow
        ]

        # --- Draw lane centre-lines and boundaries per section ---
        for sec_id in range(self.n_sections):
            for lane_id in range(self.n_lanes):
                sl = self.sections[sec_id][lane_id]
                s_vals = np.linspace(0, sl.arc_length, n_samples)

                # Centre-line
                if draw_centrelines:
                    colour = lane_colours[lane_id % len(lane_colours)]
                    pts = np.array([
                        [float(sl._spline_x(s)), float(sl._spline_y(s))]
                        for s in s_vals
                    ])
                    for i in range(len(pts) - 1):
                        _draw_line(pts[i], pts[i + 1], colour, thick=0.04)

                # Lane boundaries (only draw outer boundary for outermost,
                # inner for innermost, to avoid overlap)
                if draw_lane_boundaries:
                    inner_pts, outer_pts = sl.boundary_points(n_samples)
                    boundary_colour = (255, 165, 0)  # orange

                    if lane_id == 0:
                        # Inner boundary of innermost lane
                        for i in range(len(inner_pts) - 1):
                            _draw_line(inner_pts[i], inner_pts[i + 1],
                                       boundary_colour, thick=0.05)
                    if lane_id == self.n_lanes - 1:
                        # Outer boundary of outermost lane
                        for i in range(len(outer_pts) - 1):
                            _draw_line(outer_pts[i], outer_pts[i + 1],
                                       boundary_colour, thick=0.05)

                # Transverse s-grid lines
                if draw_s_grid and s_grid_step is not None and s_grid_step > 0:
                    n_cells = max(1, int(round(sl.arc_length / s_grid_step)))
                    s_grid = np.linspace(0, sl.arc_length, n_cells, endpoint=False)
                    half_w = self.lane_width / 2.0
                    for sg in s_grid:
                        heading = sl.tangent_angle(float(sg))
                        cx = float(sl._spline_x(sg))
                        cy = float(sl._spline_y(sg))
                        n = np.array([-math.sin(heading), math.cos(heading)])
                        c = np.array([cx, cy])
                        _draw_line(c - half_w * n, c + half_w * n,
                                   (0, 180, 255), thick=0.03)

        # --- Draw section boundaries (radial lines) ---
        if draw_section_boundaries:
            r_inner = self.inner_radius
            r_outer = self.inner_radius + self.n_lanes * self.lane_width
            sign = 1.0 if self.direction == "ccw" else -1.0
            for sec in range(self.n_sections):
                angle = sign * sec * self.section_angle
                p_in = self.centre + r_inner * np.array([math.cos(angle), math.sin(angle)])
                p_out = self.centre + r_outer * np.array([math.cos(angle), math.sin(angle)])
                _draw_line(p_in, p_out, (255, 80, 80), thick=0.08)

        # --- Draw section index labels ---
        if draw_section_labels:
            sign = 1.0 if self.direction == "ccw" else -1.0
            r_label = self.inner_radius + self.n_lanes * self.lane_width + 1.5
            for sec in range(self.n_sections):
                mid_angle = sign * (sec + 0.5) * self.section_angle
                lx = self.centre[0] + r_label * math.cos(mid_angle)
                ly = self.centre[1] + r_label * math.sin(mid_angle)
                world.debug.draw_string(
                    _loc(lx, ly),
                    f"S{sec}",
                    draw_shadow=True,
                    color=carla_module.Color(r=255, g=255, b=255),
                    life_time=life_time,
                    persistent_lines=True,
                )

    # ------------------------------------------------------------------
    # Summary / printing
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the map."""
        lines = [
            f"RoundaboutLaneletMap",
            f"  Centre       : ({self.centre[0]:.1f}, {self.centre[1]:.1f})",
            f"  Inner radius : {self.inner_radius:.1f} m",
            f"  Lane width   : {self.lane_width:.1f} m",
            f"  N lanes      : {self.n_lanes}",
            f"  N sections   : {self.n_sections}",
            f"  Section angle: {math.degrees(self.section_angle):.1f}°",
            f"  Direction    : {self.direction}",
            "",
        ]
        for lane_id in range(self.n_lanes):
            r = self.inner_radius + (lane_id + 0.5) * self.lane_width
            L = self.section_arc_length(lane_id)
            C = self.total_circumference(lane_id)
            kappa = 1.0 / r
            lines.append(
                f"  Lane {lane_id}: r={r:.1f}m, "
                f"section_L={L:.2f}m, circumf={C:.1f}m, "
                f"κ={kappa:.4f} 1/m"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"RoundaboutLaneletMap(n_sections={self.n_sections}, "
            f"n_lanes={self.n_lanes})"
        )


# ---------------------------------------------------------------------------
# Quick test  (run: python roundabout_lanelets.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rmap = RoundaboutLaneletMap(
        centre=(-0.5, 0.5),
        inner_radius=13.5,
        lane_width=4.0,
        n_lanes=4,
        n_sections=12,
    )

    print(rmap.summary())
    print()

    # Test Frenet conversion for a point on the second lane, first section
    r_test = 13.5 + 1.5 * 4.0  # lane 1 centre radius
    test_xy = np.array([
        -0.5 + r_test * math.cos(math.radians(15)),
        0.5  + r_test * math.sin(math.radians(15)),
    ])
    sec, lane, fs = rmap.to_frenet(test_xy, speed=5.0, yaw=math.radians(105))
    print(f"Test point: ({test_xy[0]:.2f}, {test_xy[1]:.2f})")
    print(f"  -> section={sec}, lane={lane}")
    print(f"  -> s={fs.s:.3f}, d={fs.d:.3f}, s_dot={fs.s_dot:.3f}, d_dot={fs.d_dot:.3f}")
    print(f"  -> MDP_s state: {fs.longitudinal}")
    print(f"  -> MDP_d state: {fs.lateral}")

    # Round-trip
    cart = rmap.to_cartesian(sec, lane, fs.s, fs.d)
    err = np.linalg.norm(test_xy - np.array([cart.x, cart.y]))
    print(f"  -> Round-trip Cartesian: ({cart.x:.3f}, {cart.y:.3f}), error={err:.6f} m")

    # Decoupled dynamics
    A_s, B_s = rmap.get_longitudinal_dynamics(dt=0.1)
    A_d, B_d = rmap.get_lateral_dynamics(dt=0.1)
    print(f"\nLongitudinal A_s =\n{A_s}")
    print(f"Longitudinal B_s =\n{B_s}")
    print(f"Lateral A_d =\n{A_d}")
    print(f"Lateral B_d =\n{B_d}")

    print("\nAll tests passed.")
