"""
main_roundabout.py
==================
Roundabout scenario with single-policy DP + MPC tracking.

By default runs a **single ego vehicle** with pedestrian.  Pass
``--multi-agent`` to add an opponent vehicle controlled by the same
DFATree policy.

Architecture
------------
    ┌───────────────┐     ┌──────────────────────┐     ┌───────────────┐
    │ Roundabout    │────▶│ DFATree DP           │────▶│ MPC Tracker   │
    │ Lanelet Map   │     │ (ego [+ opp] + ped)  │     │ (per vehicle) │
    └───────────────┘     └──────────────────────┘     └───────────────┘
"""

import argparse
import math
from typing import Optional, Tuple

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
from abstraction.roundabout_abstraction import (
    LaneletGraph,
    build_label_matrix,
    build_state_cost,
    build_action_cost,
    build_pedestrian_chain,
)

from decision.roundabout_dfa import RoundaboutDFA
from decision.maker_roundabout_dp import RoundaboutDPDecisionMaker
from abstraction.roundabout_abstraction import LaneletGraph as _LG


# ---------------------------------------------------------------------------
# MPC reference-trajectory builder (Frenet profiler)
# ---------------------------------------------------------------------------

# Action constants (must match LaneletGraph)
_ACT_ADVANCE  = _LG.ACTION_ADVANCE
_ACT_LANE_IN  = _LG.ACTION_LANE_IN
_ACT_LANE_OUT = _LG.ACTION_LANE_OUT
_ACT_YIELD    = _LG.ACTION_YIELD


def build_mpc_reference_lanelet(
    action: int,
    rmap,
    s0: float,
    d0: float,
    v0: float,
    section: int,
    lane: int,
    mpc_dt: float,
    mpc_horizon: int,
    lane_width: float,
    v_cruise: float = 8.0,
    lam: float = 2.5,
    K_lc: int = 15,
    a_brake: float = 3.0,
) -> Tuple[np.ndarray, int, int]:
    """
    Generate a smooth MPC reference trajectory from a discrete action.

    Parameters
    ----------
    action       : discrete action index (advance / lane_in / lane_out / yield)
    rmap         : RoundaboutLaneletMap
    s0, d0       : current Frenet state within *lane*
    v0           : current speed [m/s]
    section      : current section index
    lane         : current lane index
    mpc_dt       : MPC time step [s]
    mpc_horizon  : number of MPC steps
    lane_width   : width of one lane [m]
    v_cruise     : cruise speed for advance [m/s]
    lam          : exponential centering rate (d → 0)
    K_lc         : number of steps for a full lane-change cosine profile
    a_brake      : braking deceleration for yield [m/s²]

    Returns
    -------
    ref          : ndarray (4, mpc_horizon)  [x, y, v, yaw]
    final_section: section index after the rollout
    final_lane   : lane index after the rollout (may change on lane_in/out)
    """
    ref = np.zeros((4, mpc_horizon), dtype=float)

    s = s0
    d = d0
    v = v0
    sec = section
    ln = lane
    sl = rmap.get_section_lanelet(sec, ln)

    # Determine lateral target for lane changes
    if action == _ACT_LANE_IN and ln > 0:
        d_target = d0 - lane_width      # shift one lane inward in current frame
        target_lane = ln - 1
    elif action == _ACT_LANE_OUT and ln < rmap.n_lanes - 1:
        d_target = d0 + lane_width      # shift one lane outward in current frame
        target_lane = ln + 1
    else:
        d_target = 0.0
        target_lane = ln

    lane_change_done = False

    for k in range(mpc_horizon):
        # --- lateral profile ---
        if action in (_ACT_LANE_IN, _ACT_LANE_OUT) and not lane_change_done:
            # cosine transition toward d_target
            progress = min(k / max(K_lc - 1, 1), 1.0)
            d = d0 + (d_target - d0) * 0.5 * (1.0 - math.cos(math.pi * progress))
            if progress >= 1.0:
                # Switch reference lane and reset d in new frame
                ln = target_lane
                sl = rmap.get_section_lanelet(sec, ln)
                d = 0.0
                d0 = 0.0
                d_target = 0.0
                lane_change_done = True
        else:
            # exponential centering: d(k) = d_prev * exp(-lam * dt)
            d = d * math.exp(-lam * mpc_dt)

        # --- longitudinal profile ---
        if action == _ACT_YIELD:
            v = max(v - a_brake * mpc_dt, 0.0)
        else:
            # ramp toward cruise speed
            v = v + min(v_cruise - v, 2.0 * mpc_dt)
            v = max(v, 0.5)

        s += v * mpc_dt

        # --- section wrapping ---
        while s >= sl.arc_length:
            s -= sl.arc_length
            sec = rmap.next_section(sec)
            sl = rmap.get_section_lanelet(sec, ln)
        while s < 0:
            sec = rmap.prev_section(sec)
            sl = rmap.get_section_lanelet(sec, ln)
            s += sl.arc_length

        # --- Frenet → Cartesian ---
        cart = sl.to_cartesian(s, d)
        ref[0, k] = cart.x
        ref[1, k] = cart.y
        ref[2, k] = max(v, 0.5)
        ref[3, k] = cart.heading

    return ref, sec, ln


# ===========================================================================
# Main
# ===========================================================================

def main():
    argparser = argparse.ArgumentParser(description='Roundabout DP + MPC')
    argparser.add_argument('--host', default='127.0.0.1', help='Host IP')
    argparser.add_argument('-p', '--port', default=2000, type=int)
    argparser.add_argument('--multi-agent', action='store_true',
                           help='Add opponent vehicle (default: ego only)')
    args = argparser.parse_args()
    running = True
    env = None
    multi_agent = args.multi_agent

    # =====================================================================
    #  CONFIGURATION
    # =====================================================================
    # Geometry
    CENTRE       = (-0.5, 0.5)
    INNER_RADIUS = 13.5
    LANE_WIDTH   = 4.0
    N_LANES      = 4
    N_SECTIONS   = 12
    DIRECTION    = "cw"
    DRIVABLE_LANES = (1, 2)

    # Vehicle lanes
    EGO_LANE     = 2
    OPP_LANE     = 1

    # DP parameters
    GAMMA        = 0.5
    PROCESS_NOISE = 0.05

    # DFA tree solver
    N_TREE_ITERS  = 3
    N_GROW        = 2
    N_VI_PER_ITER = 10

    # Pedestrian
    N_P          = 8
    P_MOVE       = 0.3
    PED_WAIT_AT_EDGE = 1.0

    # Cost tuning
    NON_DRIVABLE_PENALTY_EGO = 50.0
    NON_DRIVABLE_PENALTY_OPP = 50.0
    ADVANCE_REWARD    = -1.0
    LANE_CHANGE_COST  = 0.5

    # DFA risk costs
    DFA_COST_ND   = 40.0
    DFA_COST_PED  = 500.0
    DFA_COST_COLL = 300.0

    # Collision detection
    COLLISION_RADIUS = 1.0

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

        # =================================================================
        #  2. CARLA ENVIRONMENT
        # =================================================================
        # Ego spawn
        _orig_spawn_ego = np.array([-2.1, 20.2])
        start_section_ego = rmap._identify_section(_orig_spawn_ego)
        sl_ego = rmap.get_section_lanelet(start_section_ego, EGO_LANE)
        cart_ego = sl_ego.to_cartesian(0.5 * sl_ego.arc_length, 0.0)
        ego_sp = carla.Transform(
            carla.Location(x=cart_ego.x, y=cart_ego.y, z=0.3),
            carla.Rotation(yaw=math.degrees(cart_ego.heading)),
        )
        print(f"Ego spawn: section={start_section_ego}, "
              f"x={cart_ego.x:.2f}, y={cart_ego.y:.2f}")

        env = Environment(args, ego_transform=ego_sp)
        env.world.tick()

        spectator = env.world.get_spectator()
        spectator.set_transform(
            carla.Transform(carla.Location(z=50), carla.Rotation(pitch=-90))
        )

        # Opponent (only in multi-agent mode)
        opp_car = None
        opp_model = None
        opp_controller = None

        if multi_agent:
            opp_section = (start_section_ego + 6) % N_SECTIONS
            sl_opp = rmap.get_section_lanelet(opp_section, OPP_LANE)
            cart_opp = sl_opp.to_cartesian(0.5 * sl_opp.arc_length, 0.0)
            opp_sp = carla.Transform(
                carla.Location(x=cart_opp.x, y=cart_opp.y, z=0.3),
                carla.Rotation(yaw=math.degrees(cart_opp.heading)),
            )
            opp_car = env.add_car(opp_sp, "200, 50, 50")
            print(f"Opp spawn: section={opp_section}, "
                  f"x={cart_opp.x:.2f}, y={cart_opp.y:.2f}")
            env.world.tick()

        # Pedestrian on outer edge
        PED_SECTION = (start_section_ego + 3) % N_SECTIONS
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
        #  3. VEHICLE MODELS + MPC CONTROLLERS
        # =================================================================
        origin = carla.Location(x=CENTRE[0], y=CENTRE[1], z=0.2)

        ego_model = model.Vehicle(env.ego_car, env.dt, origin)
        ego_controller = MPC_controller(ego_model)

        if opp_car is not None:
            opp_model = model.Vehicle(opp_car, env.dt, origin)
            opp_controller = MPC_controller(opp_model)

        MPC_DT      = ego_controller.dt
        MPC_HORIZON = ego_controller.horizon

        rmap.draw_in_carla(env.world, z=0.2, life_time=120.0,
                           draw_s_grid=False, draw_section_labels=True)

        # =================================================================
        #  4. OFFLINE: BUILD ABSTRACTION + SOLVE DFATree
        # =================================================================
        graph = LaneletGraph(
            n_sections=N_SECTIONS,
            n_lanes=N_LANES,
            drivable_lanes=list(DRIVABLE_LANES),
            inner_radius=INNER_RADIUS,
            lane_width=LANE_WIDTH,
        )

        # Transition matrix (identical dynamics for all vehicles)
        P_vehicle = graph.build_transition_matrix(process_noise=PROCESS_NOISE)

        # Labels and costs
        L_vehicle = build_label_matrix(graph, n_letters=4)
        state_cost_ego = build_state_cost(graph, NON_DRIVABLE_PENALTY_EGO)
        action_cost_ego = build_action_cost(ADVANCE_REWARD, LANE_CHANGE_COST)

        # Pedestrian chain
        ped_data = None
        L_p = None
        cost_p = None
        if N_P >= 2:
            ped_data = build_pedestrian_chain(N_P, LANE_WIDTH, N_LANES, P_MOVE)

        # Assemble abstraction dict
        abs_data = {
            'n_lanelets':       graph.n_lanelets,
            'P_ego':            P_vehicle,
            'L_ego':            L_vehicle,
            'state_cost_ego':   state_cost_ego,
            'action_cost_ego':  action_cost_ego,
            'ped_data':         ped_data,
            'L_p':              L_p,
            'cost_p':           cost_p,
        }

        if multi_agent:
            state_cost_opp = build_state_cost(graph, NON_DRIVABLE_PENALTY_OPP)
            action_cost_opp = build_action_cost(ADVANCE_REWARD, LANE_CHANGE_COST)
            abs_data['P_opp']           = P_vehicle
            abs_data['L_opp']           = L_vehicle
            abs_data['state_cost_opp']  = state_cost_opp
            abs_data['action_cost_opp'] = action_cost_opp

        # DFA
        dfa = RoundaboutDFA(
            cost_non_drivable=DFA_COST_ND,
            cost_pedestrian=DFA_COST_PED,
            cost_collision=DFA_COST_COLL,
        )
        print(dfa.summary())

        # Decision maker (single tree)
        maker = RoundaboutDPDecisionMaker(
            dfa=dfa,
            abs_data=abs_data,
            gamma=GAMMA,
            n_tree_iters=N_TREE_ITERS,
            n_vi_per_iter=N_VI_PER_ITER,
            n_grow=N_GROW,
        )

        # =================================================================
        #  5. MAIN SIMULATION LOOP
        # =================================================================
        print("\n========== Starting simulation loop ==========\n")
        step = 0
        ego_lane = EGO_LANE
        opp_lane = OPP_LANE
        LOG_INTERVAL = 50

        _ACTION_NAMES = ['advance', 'lane_in', 'lane_out', 'yield']

        while running:
            step += 1
            env.world.tick()

            # -- 1. Update vehicle models --
            ego_model.update()
            if opp_model is not None:
                opp_model.update()

            # Update pedestrian patrol
            env.update_pedestrian_patrol(
                centre=CENTRE,
                inner_radius=PED_R_INNER,
                outer_radius=PED_R_OUTER,
                section_angle=2 * math.pi / N_SECTIONS,
                wait_at_outer=PED_WAIT_AT_EDGE,
            )

            # -- 2. Project vehicles to Frenet / lanelets --
            ego_xy = np.array([ego_model.x, ego_model.y])
            ego_v = ego_model.v
            ego_yaw = ego_model.yaw

            sec_ego, lane_ego, frenet_ego = rmap.to_frenet(
                ego_xy, speed=ego_v, yaw=ego_yaw,
            )
            s_ego = frenet_ego.s
            d_ego = frenet_ego.d

            # Snap to nearest drivable lane if off-road
            if lane_ego not in DRIVABLE_LANES:
                lane_ego = min(DRIVABLE_LANES, key=lambda l: abs(l - lane_ego))
                sl_snap = rmap.get_section_lanelet(sec_ego, lane_ego)
                frenet_ego = sl_snap.to_frenet(ego_xy, speed=ego_v, yaw=ego_yaw)
                s_ego = frenet_ego.s
                d_ego = frenet_ego.d
            ego_lane = lane_ego
            ego_idx = graph.to_index(sec_ego, ego_lane)

            # Opponent Frenet (if present)
            opp_idx = None
            sec_opp = None
            s_opp = 0.0
            d_opp = 0.0
            v_opp = 0.0

            if opp_model is not None:
                opp_xy = np.array([opp_model.x, opp_model.y])
                v_opp = opp_model.v
                opp_yaw = opp_model.yaw

                sec_opp, lane_opp_det, frenet_opp = rmap.to_frenet(
                    opp_xy, speed=v_opp, yaw=opp_yaw,
                )
                s_opp = frenet_opp.s
                d_opp = frenet_opp.d

                if lane_opp_det not in DRIVABLE_LANES:
                    lane_opp_det = min(DRIVABLE_LANES,
                                       key=lambda l: abs(l - lane_opp_det))
                    sl_snap_o = rmap.get_section_lanelet(sec_opp, lane_opp_det)
                    frenet_opp = sl_snap_o.to_frenet(
                        opp_xy, speed=v_opp, yaw=opp_yaw)
                    s_opp = frenet_opp.s
                    d_opp = frenet_opp.d
                opp_lane = lane_opp_det
                opp_idx = graph.to_index(sec_opp, opp_lane)

            # -- 3. Update DFA state --
            vehicle_indices = [ego_idx]
            if opp_idx is not None:
                vehicle_indices.append(opp_idx)

            # Pedestrian lanelet (if visible)
            ped_lanelet_idx = None
            if env.pedestrian is not None:
                ped_loc = env.pedestrian.get_location()
                ped_dx = ped_loc.x - CENTRE[0]
                ped_dy = ped_loc.y - CENTRE[1]
                ped_r = math.hypot(ped_dx, ped_dy)
                ped_lane_id = int((ped_r - INNER_RADIUS) / LANE_WIDTH)
                ped_lane_id = max(0, min(N_LANES - 1, ped_lane_id))
                ped_section = rmap._identify_section(
                    np.array([ped_loc.x, ped_loc.y]))
                ped_lanelet_idx = graph.to_index(ped_section, ped_lane_id)

            dfa_label = dfa.classify_joint_state(
                vehicle_lanelet_indices=vehicle_indices,
                graph=graph,
                ped_lanelet_idx=ped_lanelet_idx,
            )
            maker.update_dfa_state(dfa_label)

            if maker.q_current == dfa.sink:
                print(f"[step {step}] DFA entered fail state "
                      f"(label='{dfa_label}'). Stopping.")
                break

            # -- 4. Query policy --
            action_ego, action_opp = maker.get_action(
                ego_lanelet=ego_idx,
                opp_lanelet=opp_idx,
            )

            # -- 5. Build MPC references --
            ref_ego, _, _ = build_mpc_reference_lanelet(
                action=action_ego,
                rmap=rmap,
                s0=s_ego, d0=d_ego, v0=ego_v,
                section=sec_ego, lane=ego_lane,
                mpc_dt=MPC_DT, mpc_horizon=MPC_HORIZON,
                lane_width=LANE_WIDTH,
            )

            ref_opp = None
            if opp_model is not None and action_opp is not None:
                ref_opp, _, _ = build_mpc_reference_lanelet(
                    action=action_opp,
                    rmap=rmap,
                    s0=s_opp, d0=d_opp, v0=v_opp,
                    section=sec_opp, lane=opp_lane,
                    mpc_dt=MPC_DT, mpc_horizon=MPC_HORIZON,
                    lane_width=LANE_WIDTH,
                )

            # -- 6. Solve MPC and apply controls --
            try:
                ctrl_ego = ego_controller.solve_trajectory(ref_ego)
                env.ego_car.apply_control(ctrl_ego)
            except Exception as e:
                print(f"[warning] MPC Ego failed: {e}. Applying hard brake.")
                env.ego_car.apply_control(
                    carla.VehicleControl(brake=1.0, throttle=0.0))

            if opp_controller is not None and ref_opp is not None:
                try:
                    ctrl_opp = opp_controller.solve_trajectory(ref_opp)
                    opp_car.apply_control(ctrl_opp)
                except Exception as e:
                    print(f"[warning] MPC Opp failed: {e}. Applying hard brake.")
                    opp_car.apply_control(
                        carla.VehicleControl(brake=1.0, throttle=0.0))

            # -- Logging --
            if step % LOG_INTERVAL == 0:
                msg = (f"[step {step:05d}] ego=({sec_ego},{ego_lane}) "
                       f"act={_ACTION_NAMES[action_ego]}")
                if action_opp is not None:
                    msg += (f"  opp=({sec_opp},{opp_lane}) "
                            f"act={_ACTION_NAMES[action_opp]}")
                msg += f"  dfa={dfa_label}"
                print(msg)

    finally:
        if env is not None:
            env.__del__()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Exit by user')
