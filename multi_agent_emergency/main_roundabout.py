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

  2. **Safety filter**  (online, crash-cost hierarchy)
     Monitors the ego car and overrides speed / lane when danger
     is detected.  If a collision occurs, the simulation terminates.

  3. **MPC tracking controller**  (Cartesian reference from DP policy)

Architecture
------------
    ┌───────────────┐     ┌────────────────────┐     ┌──────────────┐
    │ Roundabout     │────▶│ DFATree DP         │────▶│ MPC Tracker  │
    │ Lanelet Map    │     │ (offline, 1 tree)  │     │ (CARLA)      │
    └───────────────┘     └────────────────────┘     └──────────────┘
          ▲                        │                      ▲
          │  Frenet (s,d)         │ (v_s,v_d) policy     │
          └────────────────────────┘                      │
                                                          │
                              ┌──────────────┐            │
                              │ Safety Filter │────────────┘
                              │ (crash costs) │  override speed/lane
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
from abstraction.roundabout_abstraction import build_abstraction, LaneletGraph

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
    d_ref: float = 0.0,
) -> np.ndarray:
    """
    Roll out the DP policy at MPC resolution and convert each
    Frenet waypoint to Cartesian.

    Returns
    -------
    ref : np.ndarray (4, mpc_horizon)   [x, y, v, yaw]
    """
    ref = np.zeros((4, mpc_horizon), dtype=float)
    s = s0
    sec = section
    sl = rmap.get_section_lanelet(sec, lane)

    for k in range(mpc_horizon):
        v_s, v_d = action_fn(s, d0)
        s += v_s * mpc_dt

        while s >= sl.arc_length:
            s -= sl.arc_length
            sec = rmap.next_section(sec)
            sl = rmap.get_section_lanelet(sec, lane)
        while s < 0:
            sec = rmap.prev_section(sec)
            sl = rmap.get_section_lanelet(sec, lane)
            s += sl.arc_length

        cart = sl.to_cartesian(s, d_ref)
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
    PED_PENALTY  = 50.0
    PED_WAIT_AT_EDGE = 1.0

    # DP parameters
    DT_DP        = 0.05
    GAMMA        = 0.5
    K_SPEED      = 0.5
    K_LAT        = 0.1
    ROAD_REWARD  = -1.0
    EDGE_PENALTY = 50.0
    DRIVABLE_LANES = (1, 2)

    # DFA tree solver
    N_TREE_ITERS  = 3
    N_GROW        = 2
    N_VI_PER_ITER = 10

    # Safety filter
    SF_WARN_DIST     = 12.0       # pedestrian warning distance [m]
    SF_BRAKE_DIST    = 6.0        # pedestrian brake distance [m]
    SF_CAUTION_FACTOR = 0.4
    COLLISION_RADIUS = 1.5        # collision detection radius [m]

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
        abs_data = build_abstraction(
            rmap=rmap,
            ref_lane=DRIVE_LANE,
            drivable_lanes=DRIVABLE_LANES,
            dt=DT_DP,
            N_s=N_S,
            N_d=N_D,
            N_p=N_P,
            v_s_max=V_S_MAX,
            v_d_max=V_D_MAX,
            n_speed_levels_s=N_SPEED_S,
            n_speed_levels_d=N_SPEED_D,
            k_speed=K_SPEED,
            k_lat=K_LAT,
            road_reward=ROAD_REWARD,
            boundary_penalty=EDGE_PENALTY,
            p_move=P_MOVE,
            ped_on_road_penalty=PED_PENALTY,
            n_letters=3,
        )

        dfa = RoundaboutDFA()
        print(dfa.summary())

        maker = RoundaboutDPDecisionMaker(
            dfa=dfa,
            abs_data=abs_data,
            gamma=GAMMA,
            n_tree_iters=N_TREE_ITERS,
            n_vi_per_iter=N_VI_PER_ITER,
            n_grow=N_GROW,
        )

        graph = abs_data['graph']

        # =================================================================
        #  5. SAFETY FILTER
        # =================================================================
        safety_filter = SafetyFilter(
            graph=graph,
            lane_width=LANE_WIDTH,
            n_lanes=N_LANES,
            drivable_lanes=DRIVABLE_LANES,
            warn_distance=SF_WARN_DIST,
            brake_distance=SF_BRAKE_DIST,
            caution_speed_factor=SF_CAUTION_FACTOR,
        )
        print(f"\n[SafetyFilter] Initialised "
              f"(warn={SF_WARN_DIST}m, brake={SF_BRAKE_DIST}m)")

        # ---------------------------------------------------------------
        # Pedestrian distance helper
        # ---------------------------------------------------------------
        PED_ANGLE_AHEAD  = math.radians(45)
        PED_ANGLE_BEHIND = math.radians(8)

        def _get_ped_info(ego_xy: np.ndarray):
            """
            Returns (ped_distance, ped_lane) or (None, None).

            ped_distance : s-direction distance [m]
            ped_lane     : lane index the pedestrian is on
            """
            if env.pedestrian is None:
                return None, None
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
                return None, None

            # Compute approximate s-distance
            r_ego = math.sqrt(dx_e * dx_e + dy_e * dy_e)
            arc_dist = r_ego * abs(signed_ang)

            # Determine which lane the ped is on
            ped_lane = int((r_ped - INNER_RADIUS) / LANE_WIDTH)
            ped_lane = max(0, min(N_LANES - 1, ped_lane))

            return arc_dist, ped_lane

        # =================================================================
        #  6. MAIN SIMULATION LOOP
        # =================================================================
        print("\n========== Starting simulation loop ==========\n")
        step = 0
        current_lane = DRIVE_LANE

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

            # Snap to nearest drivable lane if off-road
            if lane_id not in DRIVABLE_LANES:
                lane_id = min(DRIVABLE_LANES,
                              key=lambda l: abs(l - lane_id))
            current_lane = lane_id

            # --- Pedestrian info ---
            ped_dist, ped_lane = _get_ped_info(ego_xy)

            # --- Collision detection ---
            crash_msg = safety_filter.check_collision(
                d_ego, current_lane, ped_dist, ped_lane,
                collision_radius=COLLISION_RADIUS,
            )
            if crash_msg is not None:
                raise CrashError(crash_msg)

            # --- Safety filter evaluation ---
            risk_level, sf_info = safety_filter.evaluate(
                d_ego, current_lane, ped_dist, ped_lane,
            )

            # --- DP policy lookup ---
            v_s, v_d = maker.get_action(s_ego, d_ego)

            # --- Safety filter overrides ---
            v_s_safe = safety_filter.filter_speed(v_s, risk_level)
            target_lane = safety_filter.suggest_lane(
                current_lane, risk_level, ped_lane,
            )

            # dp_action for MPC reference (uses filtered speed)
            def dp_action_filtered(s: float, d: float):
                _, v_d_nom = maker.get_action(s, d)
                return v_s_safe, v_d_nom

            # --- Build MPC reference trajectory ---
            ref_traj = build_mpc_reference(
                action_fn=dp_action_filtered,
                rmap=rmap,
                s0=s_ego, d0=d_ego,
                section=sec_id, lane=target_lane,
                mpc_dt=MPC_DT,
                mpc_horizon=MPC_HORIZON,
                d_ref=0.0,
            )

            # --- MPC tracking ---
            try:
                control_cmd = ego_controller.solve_trajectory(ref_traj)
                if risk_level == RiskLevel.BRAKE:
                    control_cmd.throttle = 0.0
                    control_cmd.brake = 1.0
                env.ego_car.apply_control(control_cmd)
            except Exception as e:
                try:
                    target_pt = (ref_traj[0, 0], ref_traj[1, 0], ref_traj[3, 0])
                    speed = max(v_s_safe, 0.5)
                    control_cmd = ego_controller.solve(target_pt, speed)
                    if risk_level == RiskLevel.BRAKE:
                        control_cmd.throttle = 0.0
                        control_cmd.brake = 1.0
                    env.ego_car.apply_control(control_cmd)
                except Exception:
                    if risk_level == RiskLevel.BRAKE:
                        env.ego_car.apply_control(
                            carla.VehicleControl(brake=1.0, throttle=0.0))
                    else:
                        env.ego_car.apply_control(
                            carla.VehicleControl(brake=0.0, throttle=0.3))

            # --- Logging ---
            if step % 100 == 0:
                print(
                    f"[Step {step}] "
                    f"sec={sec_id}, lane={current_lane}→{target_lane}, "
                    f"s={s_ego:.2f}, d={d_ego:.2f}, "
                    f"v_s={v_s:.1f}→{v_s_safe:.1f}, v_d={v_d:.2f}, "
                    f"ego_v={ego_v:.1f} m/s  "
                    f"| SF:{risk_level.name}"
                )
            elif risk_level != RiskLevel.NOMINAL and step % 10 == 0:
                print(safety_filter.summary_line(risk_level, sf_info))

    except CrashError as e:
        print(f"\n{'='*60}")
        print(f"  SIMULATION TERMINATED: {e}")
        print(f"{'='*60}\n")

    finally:
        if env is not None:
            env.__del__()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Exit by user')
