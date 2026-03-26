"""
main_roundabout.py
==================
Single-car roundabout scenario using:

  1. **Single DFATree-based DP**  (offline, computed once)
     Risk-minimising decoupled value iteration.  Cost design:
       - road cells have *negative* cost → reward for driving
       - lane edges have *positive* cost → risk penalty
       - higher forward speed gets a lower (more negative) action cost
     ⟹ The car naturally drives forward and stays centred.

  2. **MPC-based safety filter**  (online, predictive risk)
      Scores MPC horizon risk and switches mode using soft/hard
      thresholds: nominal, evasive, or emergency brake.

  3. **MPC tracking controller**  (Cartesian reference from DP policy)

Architecture
------------
    ┌───────────────┐     ┌────────────────────┐     ┌───────────────┐
    │ Roundabout     │────▶│ DFATree DP         │────▶│ MPC Tracker │
    │ Lanelet Map    │     │ (offline, 1 tree)  │     │ (CARLA)      │
    └───────────────┘     └────────────────────┘     └───────────────┘
          ▲                        │                       ▲
          │  Frenet (s,d)         │ (v_s,v_d) policy       │
          └────────────────────────┘                       │
                                                           │
                              ┌──────────────┐             │
                              │ Safety Filter │────────────┘
                              │ (MPC risk)    │  mode selection
                              └──────────────┘
"""

import argparse
import time
import math
from typing import Optional

import numpy as np
import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

_LOGIC_DIR = os.path.abspath(os.path.join(script_dir, '..', '..', 'logic_auto_driving_COPY'))
if _LOGIC_DIR not in sys.path:
    sys.path.append(_LOGIC_DIR)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from scenarios.roundabout import *            # Environment, carla
import control.vehicle_model as model
from control.trackingMPC import MPC_controller

from abstraction.roundabout_lanelets import RoundaboutLaneletMap
from abstraction.roundabout_abstraction import build_abstraction, build_relative_abstraction, LaneletGraph

from decision.specification.roundabout_dfa import RoundaboutDFA
from decision.maker_roundabout_dp import RoundaboutDPDecisionMaker
from decision.safety_filter import SafetyFilter, RiskLevel, CrashError


# ---------------------------------------------------------------------------
# MPC reference-trajectory builder
# ---------------------------------------------------------------------------

def build_mpc_reference(
    action_fn,
    rmap: RoundaboutLaneletMap,
    s0: float,
    d0: float,
    section: int,
    lane: int,
    mpc_dt: float,
    mpc_horizon: int,
) -> np.ndarray:
    """
    Roll out the DP policy at MPC resolution and convert each
    Frenet waypoint to Cartesian.

    Both s and d are integrated forward using the policy's
    (v_s, v_d) output at each step.

    Returns
    -------
    ref : np.ndarray (4, mpc_horizon)   [x, y, v, yaw]
    """
    half_w = rmap.lane_width / 2.0
    ref = np.zeros((4, mpc_horizon), dtype=float)

    s = s0
    d = d0
    sec = section
    sl = rmap.get_section_lanelet(sec, lane)

    for k in range(mpc_horizon):
        # 1. Query policy at CURRENT (s, d)
        v_s, v_d = action_fn(s, d)

        # 2. Integrate BOTH states
        s += v_s * mpc_dt
        d += v_d * mpc_dt

        # 3. Clamp d to lane bounds (matches DP grid: wrap=False)
        d = float(np.clip(d, -half_w, half_w))

        # 4. Wrap s across section boundaries
        while s >= sl.arc_length:
            s -= sl.arc_length
            sec = rmap.next_section(sec)
            sl = rmap.get_section_lanelet(sec, lane)
        while s < 0:
            sec = rmap.prev_section(sec)
            sl = rmap.get_section_lanelet(sec, lane)
            s += sl.arc_length

        # 5. Convert to Cartesian using ACTUAL d
        cart = sl.to_cartesian(s, d)
        ref[0, k] = cart.x
        ref[1, k] = cart.y
        ref[2, k] = max(abs(v_s), 1.0)
        ref[3, k] = cart.heading

    return ref


# ===========================================================================
# Main
# ===========================================================================

def main():
    argparser = argparse.ArgumentParser(description='Roundabout DP + MPC')
    argparser.add_argument('--host', default='127.0.0.1', help='Host IP')
    argparser.add_argument('-p', '--port', default=2000, type=int)
    argparser.add_argument('--risk-soft', type=float, default=1,
                           help='Soft risk threshold for switching to evasive mode')
    argparser.add_argument('--risk-hard', type=float, default=1.5,
                           help='Hard risk threshold for emergency brake mode')
    argparser.add_argument('--risk-gamma', type=float, default=0.5,
                           help='Discount factor used for predictive horizon risk')
    argparser.add_argument('--risk-log-interval', type=int, default=50,
                           help='Print compact risk line every N simulation steps')
    args = argparser.parse_args()
    running = True
    env = None

    # =====================================================================
    #  CONFIGURATION
    # =====================================================================
    # Geometry
    CENTRE       = (-0.5, 0.5)
    INNER_RADIUS = 13.5
    LANE_WIDTH   = 4.0
    N_LANES      = 4
    N_SECTIONS   = 12
    DRIVE_LANE   = 2          # initial lane (drivable inner = 1, outer = 2)
    DIRECTION    = "cw"

    # DP grid
    N_S          = 10
    N_D          = 8
    V_S_MAX      = 10.0
    V_D_MAX      = 2.0
    N_SPEED_S    = 5
    N_SPEED_D    = 5

    # Pedestrian
    N_P          = 8
    P_MOVE       = 0.3
    PED_PENALTY  = 0.5
    PED_WAIT_AT_EDGE = 1.0

    # DP parameters
    DT_DP        = 0.1
    GAMMA        = 0.5
    K_SPEED      = 0.05
    K_LAT        = 0.01
    ROAD_REWARD  = -0.01
    EDGE_PENALTY = 0.5
    DRIVABLE_LANES = (1, 2)
    TRANSITION_NOISE_S = 0.08
    TRANSITION_NOISE_D = 0.08

    # DFA tree solver
    N_TREE_ITERS  = 3
    N_GROW        = 2
    N_VI_PER_ITER = 10

    # Safety filter (predictive risk)
    SF_WARN_DIST      = 20.0       # pedestrian warning distance [m]
    SF_BRAKE_DIST     = 5.0       # pedestrian brake distance [m]
    SF_CAUTION_FACTOR = 0.4
    COLLISION_RADIUS  = 1.0        # collision detection radius [m]
    RISK_SOFT_THRESH  = float(args.risk_soft)
    RISK_HARD_THRESH  = float(args.risk_hard)
    RISK_GAMMA        = float(args.risk_gamma)
    RISK_LOG_INTERVAL = max(1, int(args.risk_log_interval))

    if RISK_HARD_THRESH <= RISK_SOFT_THRESH:
        raise ValueError("--risk-hard must be greater than --risk-soft")

    try:
        # =================================================================
        #  1. ROUNDABOUT LANELET MAP
        # =================================================================
        rmap = RoundaboutLaneletMap(
            centre=CENTRE,
            inner_radius=INNER_RADIUS,
            lane_width=LANE_WIDTH,
            n_lanes=N_LANES,
            n_sections=N_SECTIONS,
            direction=DIRECTION,
        )
        print(rmap.summary())

        section_L = rmap.section_arc_length(DRIVE_LANE)
        print(f"Drive lane section arc length (lane {DRIVE_LANE}): {section_L:.2f} m")

        # Compute spawn pose
        _orig_spawn = np.array([-2.1, 20.2])
        start_section = rmap._identify_section(_orig_spawn)
        sl_start = rmap.get_section_lanelet(start_section, DRIVE_LANE)
        cart_start = sl_start.to_cartesian(0.5 * sl_start.arc_length, 0.0)
        ego_sp = carla.Transform(
            carla.Location(x=cart_start.x, y=cart_start.y, z=0.3),
            carla.Rotation(yaw=math.degrees(cart_start.heading)),
        )
        print(f"Spawn: section={start_section}, "
              f"x={cart_start.x:.2f}, y={cart_start.y:.2f}, "
              f"heading={math.degrees(cart_start.heading):.1f}°")

        # =================================================================
        #  2. CARLA ENVIRONMENT
        # =================================================================
        env = Environment(args, ego_transform=ego_sp)
        env.world.tick()

        spectator = env.world.get_spectator()
        spectator.set_transform(
            carla.Transform(carla.Location(z=50), carla.Rotation(pitch=-90))
        )

        # Spawn pedestrian on the outer edge, opposite to ego
        PED_SECTION = (start_section + 6) % N_SECTIONS
        r_ped_spawn = INNER_RADIUS + N_LANES * LANE_WIDTH
        ped_angle = -(PED_SECTION + 0.5) * (2 * math.pi / N_SECTIONS)
        ped_x = CENTRE[0] + r_ped_spawn * math.cos(ped_angle)
        ped_y = CENTRE[1] + r_ped_spawn * math.sin(ped_angle)
        env.spawn_pedestrian(
            spawn_location=carla.Location(x=ped_x, y=ped_y, z=0.5),
            speed=1.2,
        )
        PED_R_INNER = INNER_RADIUS + 1.75 * LANE_WIDTH
        PED_R_OUTER = INNER_RADIUS + N_LANES * LANE_WIDTH

        # =================================================================
        #  3. VEHICLE MODEL + MPC
        # =================================================================
        origin = carla.Location(x=CENTRE[0], y=CENTRE[1], z=0.2)
        car_model = model.Vehicle(env.ego_car, env.dt, origin)
        ego_controller = MPC_controller(car_model)

        MPC_DT      = ego_controller.dt
        MPC_HORIZON = ego_controller.horizon

        rmap.draw_in_carla(env.world, z=0.2, life_time=120.0,
                           draw_s_grid=False, draw_section_labels=True)

        # =================================================================
        #  4. OFFLINE: BUILD ABSTRACTION + SOLVE DFATree (ONCE)
        # =================================================================
        print("[DP] Building nominal abstraction...")
        abs_data_nominal = build_abstraction(
            rmap=rmap,
            ref_lane=DRIVE_LANE,
            drivable_lanes=DRIVABLE_LANES,
            dt=DT_DP,
            N_s=N_S,
            N_d=N_D,
            N_p=0, # signal: no pedestrian dimension
            v_s_max=V_S_MAX,
            v_d_max=V_D_MAX,
            n_speed_levels_s=N_SPEED_S,
            n_speed_levels_d=N_SPEED_D,
            k_speed=K_SPEED,
            k_lat=K_LAT,
            road_reward=ROAD_REWARD,
            boundary_penalty=EDGE_PENALTY,
            p_move=P_MOVE,
            process_noise_s=TRANSITION_NOISE_S,
            process_noise_d=TRANSITION_NOISE_D,
            ped_on_road_penalty=PED_PENALTY,
            n_letters=4,
        )
        
        print("[DP] Building evasive abstraction...")
        abs_data_evasive = build_relative_abstraction(
            rmap=rmap,
            dt=DT_DP,
            N_s=N_S,
            N_d=N_D,
            dist_max=120.0,
            v_s_max=V_S_MAX,
            v_d_max=V_D_MAX,
            n_speed_levels_s=N_SPEED_S,
            n_speed_levels_d=N_SPEED_D,
            k_speed=K_SPEED,
            k_lat=K_LAT,
            collision_penalty=PED_PENALTY,
            boundary_penalty=EDGE_PENALTY,
            process_noise_s=TRANSITION_NOISE_S,
            process_noise_d=TRANSITION_NOISE_D,
            n_letters=4,
        )

        dfa = RoundaboutDFA(
            cost_non_drivable=0.4,
            cost_pedestrian=0.9,
            cost_other_car=0.6,
        )
        print(dfa.summary())

        maker = RoundaboutDPDecisionMaker(
            dfa=dfa,
            abs_data_nominal=abs_data_nominal,
            abs_data_evasive=abs_data_evasive,
            gamma=GAMMA,
            n_tree_iters=N_TREE_ITERS,
            n_vi_per_iter=N_VI_PER_ITER,
            n_grow=N_GROW,
        )

        graph = abs_data_nominal['graph']

        # =================================================================
        #  5. SAFETY FILTER
        # =================================================================
        safety_filter = SafetyFilter(
            dfa=dfa,
            graph=graph,
            lane_width=LANE_WIDTH,
            n_lanes=N_LANES,
            drivable_lanes=DRIVABLE_LANES,
            warn_distance=SF_WARN_DIST,
            brake_distance=SF_BRAKE_DIST,
            caution_speed_factor=SF_CAUTION_FACTOR,
        )
        print(
            f"\n[SafetyFilter] Initialised "
            f"(warn={SF_WARN_DIST}m, brake={SF_BRAKE_DIST}m, "
            f"r_soft={RISK_SOFT_THRESH}, r_hard={RISK_HARD_THRESH}, "
            f"gamma={RISK_GAMMA})"
        )

        # ---------------------------------------------------------------
        # Pedestrian distance helper
        # ---------------------------------------------------------------
        PED_ANGLE_AHEAD  = math.radians(180)
        PED_ANGLE_BEHIND = math.radians(8)

        _prev_ped_r = None

        def _get_ped_info(ego_xy: np.ndarray):
            """
            Returns (ped_distance, ped_lane, ped_target_lane).

            ped_distance    : s-direction distance [m]
            ped_lane        : lane index the pedestrian is currently on
            ped_target_lane : lane the pedestrian is moving toward
                              (same as ped_lane if stationary / moving
                              along the lane; None if ped is moving away
                              from the road)
            """
            nonlocal _prev_ped_r
            if env.pedestrian is None:
                _prev_ped_r = None
                return None, None, None
            loc = env.pedestrian.get_location()
            dx_p = loc.x - CENTRE[0]
            dy_p = loc.y - CENTRE[1]
            r_ped = math.sqrt(dx_p * dx_p + dy_p * dy_p)

            # Angular check: only consider ped ahead
            dx_e = ego_xy[0] - CENTRE[0]
            dy_e = ego_xy[1] - CENTRE[1]
            ang_ped = math.atan2(dy_p, dx_p)
            ang_ego = math.atan2(dy_e, dx_e)
            signed_ang = math.atan2(
                math.sin(ang_ped - ang_ego),
                math.cos(ang_ped - ang_ego),
            )

            if signed_ang < -PED_ANGLE_AHEAD or signed_ang > PED_ANGLE_BEHIND:
                _prev_ped_r = r_ped
                return None, None, None

            # Compute approximate s-distance
            r_ego = math.sqrt(dx_e * dx_e + dy_e * dy_e)
            arc_dist = r_ego * abs(signed_ang)

            # Determine which lane the ped is on
            ped_lane = int((r_ped - INNER_RADIUS) / LANE_WIDTH)
            ped_lane = max(0, min(N_LANES - 1, ped_lane))

            # Determine target lane from radial velocity
            if _prev_ped_r is not None:
                dr = r_ped - _prev_ped_r
                if dr < -0.05:
                    # Moving inward → target is one lane inward
                    ped_target_lane = max(0, ped_lane - 1)
                elif dr > 0.05:
                    # Moving outward → target is one lane outward
                    ped_target_lane = min(N_LANES - 1, ped_lane + 1)
                else:
                    # Roughly stationary radially
                    ped_target_lane = ped_lane
            else:
                ped_target_lane = ped_lane

            _prev_ped_r = r_ped
            return arc_dist, ped_lane, ped_target_lane

        def _predictive_risk_from_reference(ref_traj: np.ndarray) -> tuple[float, str]:
            """
            Compute discounted MPC-horizon risk for one candidate reference.

            For this single-agent + pedestrian setup, the pedestrian is
            treated as quasi-static over the short MPC horizon.
            """
            if env.pedestrian is None:
                return 0.0, 'safe'

            ped_loc = env.pedestrian.get_location()
            ped_dx = ped_loc.x - CENTRE[0]
            ped_dy = ped_loc.y - CENTRE[1]
            ped_r = math.hypot(ped_dx, ped_dy)
            ped_lane = int((ped_r - INNER_RADIUS) / LANE_WIDTH)
            ped_lane = max(0, min(N_LANES - 1, ped_lane))
            ped_ang = math.atan2(ped_dy, ped_dx)

            stage_costs = []
            dominant_label = 'safe'
            max_stage_cost = -1.0

            for k in range(MPC_HORIZON):
                ego_x = float(ref_traj[0, k])
                ego_y = float(ref_traj[1, k])
                ego_v = float(ref_traj[2, k])
                ego_yaw = float(ref_traj[3, k])

                _, lane_k, frenet_k = rmap.to_frenet(
                    np.array([ego_x, ego_y]), speed=ego_v, yaw=ego_yaw
                )

                # Use the same geometric "ahead" check as the online monitor.
                dx_e = ego_x - CENTRE[0]
                dy_e = ego_y - CENTRE[1]
                ang_ego = math.atan2(dy_e, dx_e)
                signed_ang = math.atan2(
                    math.sin(ped_ang - ang_ego),
                    math.cos(ped_ang - ang_ego),
                )
                if signed_ang < -PED_ANGLE_AHEAD or signed_ang > PED_ANGLE_BEHIND:
                    ped_dist_k = None
                else:
                    r_ego = math.hypot(dx_e, dy_e)
                    ped_dist_k = r_ego * abs(signed_ang)

                c_k, label_k = safety_filter.stage_risk_cost(
                    d_ego=frenet_k.d,
                    current_lane=lane_k,
                    ped_distance=ped_dist_k,
                    ped_on_lane=ped_lane,
                    ped_target_lane=_ped_target_lane,
                    collision_radius=COLLISION_RADIUS,
                )
                stage_costs.append(c_k)
                if c_k > max_stage_cost:
                    max_stage_cost = c_k
                    dominant_label = label_k

            risk_pred = safety_filter.predicted_horizon_risk(
                stage_costs=stage_costs,
                gamma=RISK_GAMMA,
            )
            return risk_pred, dominant_label

        # =================================================================
        #  6. MAIN SIMULATION LOOP
        # =================================================================
        print("\n========== Starting simulation loop ==========\n")
        step = 0
        current_lane = DRIVE_LANE
        last_operation_mode = None
        risk_history = []
        mode_counts = {
            RiskLevel.NOMINAL: 0,
            RiskLevel.CAUTION: 0,
            RiskLevel.BRAKE: 0,
        }

        while running:
            step += 1
            env.world.tick()
            car_model.update()

            # Update pedestrian patrol
            env.update_pedestrian_patrol(
                centre=CENTRE,
                inner_radius=PED_R_INNER,
                outer_radius=PED_R_OUTER,
                section_angle=2 * math.pi / N_SECTIONS,
                wait_at_outer=PED_WAIT_AT_EDGE,
            )

            # --- Ego state in Frenet ---
            ego_xy = np.array([car_model.x, car_model.y])
            ego_v  = car_model.v
            ego_yaw = car_model.yaw

            sec_id, lane_id, frenet = rmap.to_frenet(
                ego_xy, speed=ego_v, yaw=ego_yaw,
            )
            s_ego = frenet.s
            d_ego = frenet.d

            # Snap to nearest drivable lane if off-road, then recompute Frenet
            # so that s_ego/d_ego are expressed relative to the snapped lane's
            # centre (not the detected non-drivable lane's centre).
            if lane_id not in DRIVABLE_LANES:
                lane_id = min(DRIVABLE_LANES,
                              key=lambda l: abs(l - lane_id))
                sl_snap = rmap.get_section_lanelet(sec_id, lane_id)
                frenet = sl_snap.to_frenet(ego_xy, speed=ego_v, yaw=ego_yaw)
                s_ego = frenet.s
                d_ego = frenet.d
            current_lane = lane_id

            # --- Pedestrian info ---
            ped_dist, _, _ped_target_lane = _get_ped_info(ego_xy)

            # Euclidean distance: single authoritative collision metric for pedestrian.
            # Arc-length (ped_dist) is only used for proximity/risk scoring below.
            ped_euclidean_dist: Optional[float] = None
            if env.pedestrian is not None:
                ped_loc = env.pedestrian.get_location()
                ped_euclidean_dist = math.hypot(
                    ego_xy[0] - ped_loc.x, ego_xy[1] - ped_loc.y
                )

            # --- Pedestrian collision check (Euclidean, no lane filter) ---
            if ped_euclidean_dist is not None and ped_euclidean_dist < COLLISION_RADIUS:
                raise CrashError("Crashed into pedestrian (label: 'pedestrian')")

            # --- Collision detection (off-road and other car hazards) ---
            # Pedestrian is excluded here; it is already handled above with the
            # Euclidean metric, which is more reliable than the arc-length estimate.
            crash_msg = safety_filter.check_collision(
                d_ego, current_lane,
                ped_distance=None, ped_on_lane=None,
                collision_radius=COLLISION_RADIUS,
            )
            if crash_msg is not None:
                raise CrashError(crash_msg)

            # --- Advance DFA state based on actual AP violation ---
            # dfa_label reflects real violations (collision / off-road),
            # NOT proximity warnings. The safety filter prevents
            # violations; the DFA only transitions on actual failure.
            ped_collision = (
                ped_euclidean_dist is not None
                and ped_euclidean_dist < COLLISION_RADIUS
            )
            
            dfa_label = dfa.classify_state(
                d_ego=d_ego,
                lane_half_width=LANE_WIDTH / 2.0,
                lane_drivable=(current_lane in DRIVABLE_LANES),
                ped_nearby=ped_collision,
                car_nearby=False, # Single car scenario
            )
            maker.update_dfa_state(dfa_label)

            # If the DFA entered its fail state, the spec is violated
            if maker.q_current == dfa.sink:
                raise CrashError(
                    f"DFA entered fail state (label='{dfa_label}')")

            # --- Candidate 1: nominal policy reference ---
            v_s_nom, v_d_nom = maker.get_action(s_ego, d_ego, policy_type='nominal')

            def _nominal_action(s: float, d: float):
                return maker.get_action(s, d, policy_type='nominal')

            ref_nominal = build_mpc_reference(
                action_fn=_nominal_action,
                rmap=rmap,
                s0=s_ego, d0=d_ego,
                section=sec_id, lane=current_lane,
                mpc_dt=MPC_DT,
                mpc_horizon=MPC_HORIZON,
            )

            risk_nominal, risk_nom_label = _predictive_risk_from_reference(ref_nominal)
            operation_mode = safety_filter.select_mode(
                predicted_risk=risk_nominal,
                soft_threshold=RISK_SOFT_THRESH,
                hard_threshold=RISK_HARD_THRESH,
            )

            selected_policy = 'nominal'
            selected_risk = risk_nominal
            selected_risk_label = risk_nom_label
            ref_traj = ref_nominal
            v_s, v_d = v_s_nom, v_d_nom

            # --- Candidate 2: evasive policy when soft threshold is exceeded ---
            if operation_mode == RiskLevel.CAUTION:
                if ped_dist is not None:
                    v_s_eva, v_d_eva = maker.get_action(
                        ped_dist, d_ego, policy_type='evasive'
                    )

                    # Stateful closure: tracks the evolving gap over the horizon
                    _eva_delta_s = ped_dist  # initial gap

                    def _evasive_action(s: float, d: float):
                        """
                        Query evasive policy at the current (delta_s, d).
                        
                        s (absolute arc-length from build_mpc_reference) is unused —
                        the evasive policy operates in relative coordinates.
                        The gap delta_s is evolved internally to match the
                        MDP's transition model: delta_s' = delta_s - v_s * dt.
                        """
                        nonlocal _eva_delta_s
                        current_gap = _eva_delta_s

                        v_s, v_d = maker.get_action(
                            current_gap, d, policy_type='evasive'
                        )

                        # Shrink gap for the next horizon step
                        # (matches _build_relative_transitions exactly)
                        _eva_delta_s = max(current_gap - v_s * MPC_DT, 0.0)

                        return v_s, v_d
                
                    ref_evasive = build_mpc_reference(
                        action_fn=_evasive_action,
                        rmap=rmap,
                        s0=s_ego, d0=d_ego,
                        section=sec_id, lane=current_lane,
                        mpc_dt=MPC_DT,
                        mpc_horizon=MPC_HORIZON,
                    )
                    risk_evasive, risk_eva_label = _predictive_risk_from_reference(ref_evasive)

                    # If evasive still predicts hard-risk, fail-safe brake.
                    if risk_evasive > RISK_HARD_THRESH:
                        operation_mode = RiskLevel.BRAKE
                    else:
                        selected_policy = 'evasive'
                        selected_risk = risk_evasive
                        selected_risk_label = risk_eva_label
                        ref_traj = ref_evasive
                        v_s, v_d = v_s_eva, v_d_eva
                else:
                    # No pedestrian estimate available; keep nominal.
                    operation_mode = RiskLevel.NOMINAL

            mode_counts[operation_mode] += 1
            risk_history.append(float(selected_risk))

            # --- MPC tracking ---
            try:
                if operation_mode == RiskLevel.BRAKE:
                    control_cmd = carla.VehicleControl(brake=1.0, throttle=0.0)
                else:
                    control_cmd = ego_controller.solve_trajectory(ref_traj)
                env.ego_car.apply_control(control_cmd)
            except Exception as e:
                try:
                    target_pt = (ref_traj[0, 0], ref_traj[1, 0], ref_traj[3, 0])
                    speed = max(v_s, 0.5)
                    if operation_mode == RiskLevel.BRAKE:
                        control_cmd = carla.VehicleControl(brake=1.0, throttle=0.0)
                    else:
                        control_cmd = ego_controller.solve(target_pt, speed)
                    env.ego_car.apply_control(control_cmd)
                except Exception:
                    if operation_mode == RiskLevel.BRAKE:
                        env.ego_car.apply_control(
                            carla.VehicleControl(brake=1.0, throttle=0.0))
                    else:
                        env.ego_car.apply_control(
                            carla.VehicleControl(brake=0.0, throttle=0.3))

            # --- Compact risk logging ---
            mode_changed = (operation_mode != last_operation_mode)
            periodic = (step % RISK_LOG_INTERVAL == 0)
            if mode_changed or periodic:
                print(
                    f"[Risk] step={step:05d} mode={operation_mode.name:<7} "
                    f"policy={selected_policy:<7} risk={selected_risk:7.1f} "
                    f"label={selected_risk_label:<11} "
                    f"thr=({RISK_SOFT_THRESH:.1f},{RISK_HARD_THRESH:.1f})"
                )
            last_operation_mode = operation_mode

    except CrashError as e:
        print(f"\n{'='*60}")
        print(f"  SIMULATION TERMINATED: {e}")
        print(f"{'='*60}\n")

    finally:
        if 'risk_history' in locals() and risk_history:
            arr = np.asarray(risk_history, dtype=float)
            p50 = float(np.percentile(arr, 50))
            p90 = float(np.percentile(arr, 90))
            p99 = float(np.percentile(arr, 99))
            suggested_soft = max(1.0, p90)
            suggested_hard = max(suggested_soft + 1.0, p99)
            total = len(risk_history)
            print("\n[RiskSummary]")
            print(f"  samples={total}")
            print(
                "  mode_counts="
                f"NOMINAL:{mode_counts[RiskLevel.NOMINAL]} "
                f"CAUTION:{mode_counts[RiskLevel.CAUTION]} "
                f"BRAKE:{mode_counts[RiskLevel.BRAKE]}"
            )
            print(
                f"  risk_percentiles: p50={p50:.1f}, p90={p90:.1f}, p99={p99:.1f}"
            )
            print(
                "  tuning_hint: "
                f"try --risk-soft {suggested_soft:.1f} --risk-hard {suggested_hard:.1f}"
            )
        if env is not None:
            env.__del__()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Exit by user')
