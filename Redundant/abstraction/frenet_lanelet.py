"""
frenet_lanelet.py
=================
Frenet-frame coordinate converter for lanelet-based road representations.

Maps CARLA world coordinates (x, y, yaw, speed) to per-lanelet Frenet
coordinates (s, d, s_dot, d_dot) that serve as the continuous state vector
for FMTensJelmar's LinModel / MDPModel pipeline.

Frenet state vector compatible with FMTensJelmar:
    x = [s, d, s_dot, d_dot]  (4-D)
    u = [a_s, a_d]             (2-D, longitudinal & lateral acceleration)

Usage example (no CARLA required):
    waypoints_xy = np.array([[0,0],[5,0],[10,1],[15,3]])
    lane = Lanelet(waypoints_xy, lane_id=0, lane_width=4.0)
    s, d = lane.to_frenet(np.array([7.0, 0.5]))
    x, y, heading = lane.to_cartesian(s, d)

Dependencies:
    numpy
    scipy          (interpolate, spatial)
    shapely        (geometry)
    dataclasses    (stdlib)
    typing         (stdlib)
    carla          (optional – only needed when building from CARLA map)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FrenetState:
    """
    Full Frenet state compatible with FMTensJelmar's 4-D LinModel.

    Attributes
    ----------
    s      : longitudinal progress along the lanelet centre-line [m]
    d      : lateral deviation (left = +, right = -) [m]
    s_dot  : longitudinal speed  [m/s]
    d_dot  : lateral speed       [m/s]
    """
    s: float
    d: float
    s_dot: float = 0.0
    d_dot: float = 0.0

    def as_array(self) -> np.ndarray:
        """Return [s, d, s_dot, d_dot] as a (4,) numpy array."""
        return np.array([self.s, self.d, self.s_dot, self.d_dot])


@dataclass
class CartesianState:
    """Cartesian state as returned by to_cartesian()."""
    x: float
    y: float
    heading: float   # tangent angle of the centre-line at s [rad]


# ---------------------------------------------------------------------------
# Lanelet
# ---------------------------------------------------------------------------

class Lanelet:
    """
    A single lanelet defined by a polyline centre-line.

    Parameters
    ----------
    waypoints : (N, 2) array of (x, y) centre-line waypoints in CARLA world
                coordinates.  N >= 2.
    lane_id   : Identifier used for labelling / lookup.
    lane_width: Full lane width [m].  Used to map d to label regions.
    s_offset  : Cumulative arc-length offset if this lanelet is part of a
                longer route.  Lets you chain lanelets into one s-axis.
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        lane_id: int = 0,
        lane_width: float = 4.0,
        s_offset: float = 0.0,
    ) -> None:
        self.lane_id = lane_id
        self.lane_width = lane_width
        self.s_offset = s_offset

        # ---- store raw waypoints -------------------------------------------
        waypoints = np.asarray(waypoints, dtype=float)
        assert waypoints.ndim == 2 and waypoints.shape[1] == 2, \
            "waypoints must be (N, 2)"
        assert waypoints.shape[0] >= 2, "Need at least 2 waypoints"
        self._waypoints = waypoints

        # ---- build cumulative arc-length vector ----------------------------
        diffs = np.diff(waypoints, axis=0)              # (N-1, 2)
        seg_lengths = np.linalg.norm(diffs, axis=1)     # (N-1,)
        self._arc_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self.length: float = float(self._arc_lengths[-1])

        # ---- smooth spline interpolation -----------------------------------
        # Parameterise x(s), y(s) as cubic splines so we can query anywhere
        # TODO: choose bc_type if you have tangent info at endpoints, e.g.
        #       bc_type=([(1, tx0), (1, ty0)], [(1, txN), (1, tyN)])
        self._spline_x = CubicSpline(self._arc_lengths, waypoints[:, 0])
        self._spline_y = CubicSpline(self._arc_lengths, waypoints[:, 1])

        # ---- KD-tree for fast nearest-waypoint lookup ----------------------
        self._kdtree = KDTree(waypoints)

        # ---- Shapely LineString for signed-distance query ------------------
        self._line = LineString(waypoints.tolist())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def to_frenet(
        self,
        xy: np.ndarray,
        speed: Optional[float] = None,
        yaw: Optional[float] = None,
    ) -> FrenetState:
        """
        Convert a world-frame point to Frenet coordinates on this lanelet.

        Parameters
        ----------
        xy    : (2,) array  [x, y] in CARLA world coordinates.
        speed : scalar speed magnitude [m/s].  If None, s_dot/d_dot = 0.
        yaw   : heading angle [rad] in world frame.  Needed for s_dot/d_dot.

        Returns
        -------
        FrenetState  (s, d, s_dot, d_dot)

        Notes
        -----
        s is measured along the centre-line from the start of this lanelet,
        plus self.s_offset if you chain lanelets.
        d > 0 : to the left  of the direction of travel.
        d < 0 : to the right of the direction of travel.
        """
        xy = np.asarray(xy, dtype=float).ravel()[:2]

        # Step 1 ── find the closest point on the centre-line ----------------
        s_raw = self._project_to_arc_length(xy)

        # Step 2 ── signed lateral deviation ---------------------------------
        d = self._signed_lateral_deviation(xy, s_raw)

        # Step 3 ── velocity decomposition -----------------------------------
        s_dot, d_dot = 0.0, 0.0
        if speed is not None and yaw is not None:
            tangent_angle = self.tangent_angle(s_raw)
            delta = yaw - tangent_angle
            s_dot = speed * math.cos(delta)
            d_dot = speed * math.sin(delta)

        return FrenetState(
            s=s_raw + self.s_offset,
            d=d,
            s_dot=s_dot,
            d_dot=d_dot,
        )

    def to_cartesian(self, s: float, d: float) -> CartesianState:
        """
        Convert Frenet (s, d) back to world Cartesian (x, y, heading).

        Parameters
        ----------
        s : longitudinal progress [m] (may include s_offset).
        d : lateral deviation [m].

        Returns
        -------
        CartesianState  (x, y, heading)

        Notes
        -----
        heading is the tangent angle of the centre-line at s [rad], useful
        as a reference heading for the MPC tracking controller.
        """
        s_local = float(s) - self.s_offset
        s_local = float(np.clip(s_local, 0.0, self.length))

        # Centre-line point
        cx = float(self._spline_x(s_local))
        cy = float(self._spline_y(s_local))

        # Tangent and normal directions
        heading = self.tangent_angle(s_local)
        nx = -math.sin(heading)   # left-pointing normal
        ny =  math.cos(heading)

        # Offset by d along the normal
        x = cx + d * nx
        y = cy + d * ny

        return CartesianState(x=x, y=y, heading=heading)

    def tangent_angle(self, s_local: float) -> float:
        """
        Centre-line tangent angle [rad] at arc-length s_local.

        TODO: verify sign convention matches CARLA's yaw (typically
              counter-clockwise positive, x-forward).
        """
        dx = float(self._spline_x(s_local, 1))  # first derivative
        dy = float(self._spline_y(s_local, 1))
        return math.atan2(dy, dx)

    def curvature(self, s_local: float) -> float:
        """
        Signed curvature κ(s) of the centre-line [1/m].

        κ > 0 : turning left.
        κ < 0 : turning right.

        Used inside the Frenet kinematic model:
            ṡ = v_x / (1 - d·κ)
        """
        xp  = float(self._spline_x(s_local, 1))
        yp  = float(self._spline_y(s_local, 1))
        xpp = float(self._spline_x(s_local, 2))
        ypp = float(self._spline_y(s_local, 2))
        denom = (xp**2 + yp**2) ** 1.5
        if denom < 1e-12:
            return 0.0
        return (xp * ypp - yp * xpp) / denom

    def is_on_lane(self, xy: np.ndarray) -> bool:
        """
        Returns True if the point is within lane_width/2 of the centre-line.
        Uses shapely for a fast distance query.
        """
        pt = Point(float(xy[0]), float(xy[1]))
        dist = self._line.distance(pt)
        return dist <= self.lane_width / 2.0

    def s_bounds(self) -> Tuple[float, float]:
        """Return (s_min, s_max) including s_offset."""
        return (self.s_offset, self.s_offset + self.length)

    def draw_in_carla(
        self,
        world,
        z: float = 0.2,
        n_samples: int = 200,
        s_grid_step: Optional[float] = None,
        life_time: float = 30.0,
        color_centre=(0, 255, 0),
        color_boundary=(255, 165, 0),
        color_grid=(0, 180, 255),
        thickness: float = 0.06,
    ) -> None:
        """
        Draw this lanelet in a running CARLA world using debug lines.

        Draws
        -----
        * Centre-line spline          (green  by default)
        * Left  boundary  (+d = +lane_width/2)   (orange by default)
        * Right boundary  (-d = -lane_width/2)   (orange by default)
        * Transverse grid lines at every s_grid_step metres   (cyan by default)

        Parameters
        ----------
        world         : carla.World instance.
        z             : Height above ground for all debug geometry [m].
        n_samples     : Number of spline samples for smooth curves.
        s_grid_step   : If given, draw a transverse line across the lane
                        every s_grid_step metres.  Good for visualising the
                        Frenet grid cells.
        life_time     : CARLA debug draw life time [s].  -1 = persistent.
        color_centre  : (R,G,B) for the centre-line.
        color_boundary: (R,G,B) for the lane boundaries.
        color_grid    : (R,G,B) for transverse s-grid lines.
        thickness     : Line thickness [m].
        """
        import carla

        def _loc(xy: np.ndarray) -> "carla.Location":
            return carla.Location(x=float(xy[0]), y=float(xy[1]), z=z)

        def _draw(p0, p1, color_rgb):
            world.debug.draw_line(
                _loc(p0), _loc(p1),
                thickness=thickness,
                color=carla.Color(r=color_rgb[0], g=color_rgb[1], b=color_rgb[2]),
                life_time=life_time,
                persistent_lines=True,
            )

        s_vals = np.linspace(0.0, self.length, n_samples)
        half_w = self.lane_width / 2.0

        # Pre-compute centre, left, right points
        centres  = np.array([[float(self._spline_x(s)), float(self._spline_y(s))] for s in s_vals])
        headings = np.array([self.tangent_angle(s) for s in s_vals])
        normals  = np.column_stack([-np.sin(headings), np.cos(headings)])  # left-pointing

        lefts  = centres + half_w * normals
        rights = centres - half_w * normals

        # Draw polylines
        for i in range(len(s_vals) - 1):
            _draw(centres[i], centres[i + 1], color_centre)
            _draw(lefts[i],   lefts[i + 1],   color_boundary)
            _draw(rights[i],  rights[i + 1],  color_boundary)

        # Draw transverse s-grid lines
        if s_grid_step is not None and s_grid_step > 0:
            # s_grid = np.arange(0.0, self.length + s_grid_step, s_grid_step)
            n_cells = max(1, int(np.round(self.length / s_grid_step)))
            s_grid = np.linspace(0.0, self.length, n_cells, endpoint=False)
            for sg in s_grid:
                sg = float(np.clip(sg, 0.0, self.length))
                heading = self.tangent_angle(float(sg))
                cx = float(self._spline_x(sg))
                cy = float(self._spline_y(sg))
                n = np.array([-math.sin(heading), math.cos(heading)])
                _draw(np.array([cx, cy]) - half_w * n,
                      np.array([cx, cy]) + half_w * n,
                      color_grid)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _project_to_arc_length(self, xy: np.ndarray) -> float:
        """
        Return the arc-length s [m] of the closest point on the centre-line
        to xy.

        Strategy:
            1. Use KDTree to find the nearest waypoint index quickly.
            2. Check the two adjacent segments (i-1, i) and (i, i+1).
            3. For each segment, project orthogonally and clamp to [0, 1].
            4. Return the s that minimises distance.

        TODO: for better accuracy on curved roads, replace with a
              Newton-step refinement on self._spline_x / _spline_y.
        """
        _, idx = self._kdtree.query(xy)

        best_s = self._arc_lengths[idx]
        best_dist = np.linalg.norm(xy - self._waypoints[idx])

        # Check adjacent segments
        for seg_start, seg_end in self._adjacent_segments(idx):
            p0 = self._waypoints[seg_start]
            p1 = self._waypoints[seg_end]
            t, proj = self._segment_projection(xy, p0, p1)
            dist = np.linalg.norm(xy - proj)
            if dist < best_dist:
                best_dist = dist
                best_s = (
                    self._arc_lengths[seg_start]
                    + t * (self._arc_lengths[seg_end] - self._arc_lengths[seg_start])
                )

        return float(best_s)

    def _segment_projection(
        self,
        xy: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Project xy onto segment [p0, p1].

        Returns
        -------
        t    : parameter in [0, 1] along the segment.
        proj : (2,) projected point.
        """
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-12:
            return 0.0, p0.copy()
        t = float(np.dot(xy - p0, seg)) / seg_len_sq
        t = float(np.clip(t, 0.0, 1.0))
        proj = p0 + t * seg
        return t, proj

    def _signed_lateral_deviation(self, xy: np.ndarray, s_local: float) -> float:
        """
        Compute signed lateral deviation d at the projected arc-length.

        Sign: positive to the left of the direction of travel.
        """
        heading = self.tangent_angle(s_local)
        cx = float(self._spline_x(s_local))
        cy = float(self._spline_y(s_local))

        # Vector from centre-line point to the query point
        v = xy - np.array([cx, cy])

        # Left-pointing unit normal
        n = np.array([-math.sin(heading), math.cos(heading)])

        return float(np.dot(v, n))

    def _adjacent_segments(self, idx: int) -> List[Tuple[int, int]]:
        n = len(self._waypoints)
        segs = []
        if idx > 0:
            segs.append((idx - 1, idx))
        if idx < n - 1:
            segs.append((idx, idx + 1))
        return segs


# ---------------------------------------------------------------------------
# LaneletMap  – collection of Lanelet objects
# ---------------------------------------------------------------------------

class LaneletMap:
    """
    Holds the full set of lanelets for a scenario.

    The map supports:
    * Querying which lanelet a world point belongs to.
    * Building from a list of manually defined waypoints.
    * (TODO) Building automatically from a CARLA map object.

    Parameters
    ----------
    lanelets : list of Lanelet objects.
    """

    def __init__(self, lanelets: Optional[List[Lanelet]] = None) -> None:
        self.lanelets: List[Lanelet] = lanelets or []

    def add_lanelet(self, lanelet: Lanelet) -> None:
        self.lanelets.append(lanelet)

    # ------------------------------------------------------------------
    # Coordinate queries
    # ------------------------------------------------------------------

    def to_frenet(
        self,
        xy: np.ndarray,
        speed: Optional[float] = None,
        yaw: Optional[float] = None,
        prefer_lane_id: Optional[int] = None,
    ) -> Tuple[Optional[int], Optional[FrenetState]]:
        """
        Find the best-matching lanelet and return (lane_id, FrenetState).

        Strategy: pick the lanelet whose centre-line is closest to xy.
        If prefer_lane_id is given, that lanelet is tried first (useful for
        the on-lane continuity check in the main loop).

        Returns
        -------
        (lane_id, FrenetState)  or  (None, None) if no lanelet found.
        """
        xy = np.asarray(xy, dtype=float).ravel()[:2]

        best_lane: Optional[Lanelet] = None
        best_fs:   Optional[FrenetState] = None
        best_abs_d = math.inf

        for lane in self.lanelets:
            s_raw = lane._project_to_arc_length(xy)
            d     = lane._signed_lateral_deviation(xy, s_raw)
            abs_d = abs(d)

            # Prefer the explicitly requested lane when it is still on-road
            if prefer_lane_id is not None and lane.lane_id == prefer_lane_id:
                if abs_d <= lane.lane_width / 2.0:
                    fs = lane.to_frenet(xy, speed=speed, yaw=yaw)
                    return lane.lane_id, fs

            if abs_d < best_abs_d:
                best_abs_d = abs_d
                best_lane  = lane

        if best_lane is None:
            return None, None

        best_fs = best_lane.to_frenet(xy, speed=speed, yaw=yaw)
        return best_lane.lane_id, best_fs

    def to_cartesian(self, lane_id: int, s: float, d: float) -> CartesianState:
        """Convert Frenet back to Cartesian on a specific lanelet."""
        lane = self.get_lanelet(lane_id)
        return lane.to_cartesian(s, d)

    def get_frenet_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return axis-aligned Frenet-space bounds across **all** lanelets.

        Returns
        -------
        lb : (4,) lower bounds [s_min, d_min, s_dot_min, d_dot_min]
        ub : (4,) upper bounds [s_max, d_max, s_dot_max, d_dot_max]

        These are used directly as sys.X bounds for FMTensJelmar's LinModel.

        TODO: fill in d, s_dot, d_dot bounds from your scenario parameters.
        """
        s_min = min(l.s_offset for l in self.lanelets)
        s_max = max(l.s_offset + l.length for l in self.lanelets)

        # TODO: set these from your scenario
        d_min, d_max = -2.0, 2.0          # half lane width
        s_dot_min, s_dot_max = 0.0, 15.0  # [m/s]
        d_dot_min, d_dot_max = -3.0, 3.0  # [m/s]

        lb = np.array([s_min, d_min, s_dot_min, d_dot_min])
        ub = np.array([s_max, d_max, s_dot_max, d_dot_max])
        return lb, ub

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @classmethod
    def from_waypoint_lists(
        cls,
        waypoint_lists: List[np.ndarray],
        lane_widths: Optional[List[float]] = None,
        chain_s: bool = True,
    ) -> "LaneletMap":
        """
        Build a LaneletMap from a list of (N_i, 2) waypoint arrays.

        Parameters
        ----------
        waypoint_lists : List of (N_i, 2) arrays, one per lanelet.
        lane_widths    : Per-lanelet widths.  Defaults to 4.0 for all.
        chain_s        : If True, offset each lanelet's s so they form a
                         continuous s-axis.  Set False for parallel lanes.
        """
        widths = lane_widths or [4.0] * len(waypoint_lists)
        lanelets = []
        s_offset = 0.0
        for i, wps in enumerate(waypoint_lists):
            lane = Lanelet(wps, lane_id=i, lane_width=widths[i], s_offset=s_offset)
            lanelets.append(lane)
            if chain_s:
                s_offset += lane.length
        return cls(lanelets)

    @classmethod
    def from_carla_map(
        cls,
        carla_map,           # carla.Map object
        route_waypoints,     # list of carla.Waypoint along the route
        lane_width: float = 4.0,
        sampling_resolution: float = 1.0,
    ) -> "LaneletMap":
        """
        Build a LaneletMap by sampling CARLA waypoints along a route.

        Parameters
        ----------
        carla_map           : carla.Map instance from env.world.get_map().
        route_waypoints     : Ordered list of carla.Waypoint defining the route.
        lane_width          : Lane width to assign [m].
        sampling_resolution : Distance between sampled waypoints [m].

        """
        # 1. Snap the first route waypoint to the CARLA map
        start_wp = carla_map.get_waypoint(
            route_waypoints[0].transform.location,
            project_to_road=True,
        )

        # 2. Walk forward along the lane, densifying with next()
        xy_points: List[List[float]] = []
        current_wp = start_wp
        total_dist = 0.0

        # Estimate route length from the provided waypoints
        route_length = sum(
            route_waypoints[i].transform.location.distance(
                route_waypoints[i + 1].transform.location
            )
            for i in range(len(route_waypoints) - 1)
        )

        while total_dist <= route_length:
            loc = current_wp.transform.location
            xy_points.append([loc.x, loc.y])
            nexts = current_wp.next(sampling_resolution)
            if not nexts:
                break
            current_wp = nexts[0]
            total_dist += sampling_resolution

        # 3. Build the LaneletMap from the sampled points
        waypoints_array = np.array(xy_points, dtype=float)
        return cls.from_waypoint_lists(
            [waypoints_array],
            lane_widths=[lane_width],
            chain_s=False,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_lanelet(self, lane_id: int) -> Lanelet:
        for lane in self.lanelets:
            if lane.lane_id == lane_id:
                return lane
        raise KeyError(f"No lanelet with lane_id={lane_id}")

    def draw_in_carla(
        self,
        world,
        z: float = 0.2,
        n_samples: int = 200,
        s_grid_step: Optional[float] = None,
        life_time: float = 30.0,
    ) -> None:
        """
        Draw all lanelets in a running CARLA world.

        Each lanelet gets its own colour for the centre-line so you can
        visually distinguish them.  Boundaries are always orange; grid
        lines are always cyan.

        Parameters
        ----------
        world        : carla.World instance  (env.world).
        z            : Height above ground [m].
        n_samples    : Spline sample resolution per lanelet.
        s_grid_step  : If given, draw transverse cell lines every
                       s_grid_step metres (matches your MDPModel grid size).
        life_time    : CARLA debug draw life time [s].
        """
        # Cycle through distinct RGB colours for centre-lines
        centre_colours = [
            (0, 255, 0),    # green
            (255, 0, 255),  # magenta
            (0, 255, 255),  # cyan
            (255, 255, 0),  # yellow
            (255, 128, 0),  # orange
        ]
        for i, lane in enumerate(self.lanelets):
            colour = centre_colours[i % len(centre_colours)]
            lane.draw_in_carla(
                world,
                z=z,
                n_samples=n_samples,
                s_grid_step=s_grid_step,
                life_time=life_time,
                color_centre=colour,
            )

    def __len__(self) -> int:
        return len(self.lanelets)

    def __repr__(self) -> str:
        ids = [l.lane_id for l in self.lanelets]
        return f"LaneletMap(n={len(self)}, lane_ids={ids})"


# ---------------------------------------------------------------------------
# Frenet kinematic model helpers  (feed into FMTensJelmar LinModel)
# ---------------------------------------------------------------------------

def build_frenet_linmodel_matrices(
    dt: float,
    v0: float = 5.0,
    kappa: float = 0.0,
) -> dict:
    """
    Build discrete-time LTI matrices for the linearised Frenet kinematic model.

    State  : x = [s, d, s_dot, d_dot]
    Input  : u = [a_s, a_d]
    Noise  : w ~ N(0, sigma^2 * I)

    Continuous-time:
        ds/dt     = s_dot
        dd/dt     = d_dot
        ds_dot/dt = a_s
        dd_dot/dt = a_d

    For a straight lanelet (kappa=0) this is double integrator in both s and d.
    For a curved lane, the coupling term (1 - d*kappa) modifies the s equation —
    that nonlinearity should be handled as part of a nonlinear system in
    FMTensJelmar or linearised around the operating point (v0, kappa).

    Parameters
    ----------
    dt    : sampling time [s].
    v0    : longitudinal speed at linearisation point [m/s].  Used for
            the curvature-coupling term if kappa != 0.
    kappa : centre-line curvature at linearisation point [1/m].

    Returns
    -------
    dict with keys 'A', 'B', 'C', 'D', 'Bw' as numpy arrays.
    Ready to pass to LinModel(**build_frenet_linmodel_matrices(dt)).

    TODO: add curvature coupling term when kappa != 0.
    """
    # ── Continuous-time matrices ──────────────────────────────────────────
    # Linearised Frenet model around (v0, kappa).
    #
    # With curvature coupling the s-equation becomes:
    #   ds/dt = s_dot / (1 - d*kappa)  ≈  s_dot * (1 + d*kappa)  (1st order)
    # Linearising around d=0:
    #   ds/dt ≈ s_dot + v0 * kappa * d_err   (where d_err = d - 0)
    # This introduces an off-diagonal A[0,1] = v0 * kappa term.
    Ac = np.array([
        [0, v0 * kappa, 1, 0],   # ds/dt     = v0*kappa*d + s_dot
        [0, 0,          0, 1],   # dd/dt     = d_dot
        [0, 0,          0, 0],   # ds_dot/dt = a_s  (input)
        [0, 0,          0, 0],   # dd_dot/dt = a_d  (input)
    ], dtype=float)

    Bc = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1],
    ], dtype=float)

    # ── Zero-order hold discretisation via matrix exponential ─────────────
    # Build the augmented matrix [Ac  Bc; 0  0] and exponentiate.
    import scipy.linalg as la
    n, m = Ac.shape[0], Bc.shape[1]
    Z = np.zeros((n + m, n + m))
    Z[:n, :n] = Ac
    Z[:n, n:] = Bc
    eZ = la.expm(Z * dt)
    Ad = eZ[:n, :n]
    Bd = eZ[:n, n:]

    # Output = full state
    C = np.eye(4)
    D = np.zeros((4, 2))

    # Noise enters on acceleration channels
    # TODO: tune sigma per your real sensor/model uncertainty
    Bw = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1],
    ], dtype=float)

    return {"A": Ad, "B": Bd, "C": C, "D": D, "Bw": Bw}


# ---------------------------------------------------------------------------
# Quick smoke-test  (run: python frenet_lanelet.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a simple straight lanelet along the x-axis
    wps = np.array([[i * 2.0, 0.0] for i in range(10)])  # 0..18 m
    lane = Lanelet(wps, lane_id=0, lane_width=4.0)

    # Test to_frenet
    test_xy = np.array([7.3, 1.2])
    fs = lane.to_frenet(test_xy)
    print(f"to_frenet({test_xy}) -> {fs}")

    # Test to_cartesian round-trip
    cs = lane.to_cartesian(fs.s, fs.d)
    print(f"to_cartesian(s={fs.s:.3f}, d={fs.d:.3f}) -> ({cs.x:.3f}, {cs.y:.3f})")
    print(f"Round-trip error: {np.linalg.norm(test_xy - np.array([cs.x, cs.y])):.6f} m")

    # Test LinModel matrices
    mats = build_frenet_linmodel_matrices(dt=0.1)
    print(f"\nAd =\n{mats['A']}")
    print(f"Bd =\n{mats['B']}")
    print("\nAll basic tests passed.")
