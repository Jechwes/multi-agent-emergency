# This is a sample Python script.
import sys
import os
import matplotlib.pyplot as plt
from utils.plot import risk_plot

# Add the directory containing your local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
local_module_dir = os.path.join(script_dir, 'decision/')
for root, dirs, files in os.walk(local_module_dir):
    if root not in sys.path:
        sys.path.append(root)

import argparse
import pygame

try:
    import numpy as np
    import sys
    from os import path as osp
except ImportError:
    raise RuntimeError('import error!')

from scenarios.intersection import *
import control.vehicle_model as model
from control.pid import *
from control.trackingMPC import MPC_controller
from perception.grid import Griding
import time
from decision.abstraction.MDP import MDP
from decision.specification.ltl_spec import Translate
from decision.abstraction.abstract import Abstraction
from decision.maker import Risk_LTL
from utils.logger import *

def traffic_light_model():
    """
    Creates a simple Markov Decision Process (MDP) for a traffic light.
    The traffic light has two states: Green ('g') and Red ('r').
    Transitions between states are probabilistic (0.8 probability to stay, 0.2 to switch).
    """
    # ---------- Traffic Light --------------------
    traffic_light = ['g', 'r']
    state_set = range(len(traffic_light))
    action_set = [0]
    transitions = np.array([[[0.8, 0.2], [0.2, 0.8]]])
    initial_state = 0
    mdp_env = MDP(state_set, action_set, transitions, traffic_light, initial_state)
    return mdp_env

def reachability_check(cur_pos, target_pos, threshold):
    """
    Checks if the current position is within a certain distance (threshold) of the target position.
    """
    if np.linalg.norm(np.array(cur_pos) - np.array(target_pos)) < threshold:
        return True
    else:
        return False


def main():

    argparser = argparse.ArgumentParser(description='Carla ArgParser practice')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    args = argparser.parse_args()
    running = True
    pygame.init()
    pygame.font.init()

    # ---------- Specification Define --------------------
    # Define Linear Temporal Logic (LTL) specifications for safety and task completion.
    # safe_spec: "Globally (G), if not Green light (~g) imply not Crossing (~c), AND Globally not Vehicle (~v), AND Globally not Obstacle (~o)"
    safe_spec = Translate("G(~g -> ~c) & G(~v) & G(~o)", AP_set=['c', 'g', 'v', 'o'])
    # safe_spec = Translate("G(~c U g) & G(~v) & G(~o)", AP_set=['c', 'g', 'v', 'o'])
    # scltl_spec: "Finally (F) reach Target (t)"
    scltl_spec = Translate("F(t)", AP_set=['t'])

    try:
        env = Environment(args) # Initialize the simulation environment (connect to CARLA, spawn actors)
        env.world.tick()

        # Define the origin point for the abstract grid system
        origin_point = carla.Location(x=82.25, y=37.5, z=0.2)
        
        # Setting for ego agent (Our autonomous car)
        # Initialize vehicle physics model and MPC controller
        ego_car_model = model.Vehicle(env.ego_car, env.dt, origin_point)
        ego_controller = MPC_controller(ego_car_model)

        # Setting for opponent agent (The other car)
        opp_car_model = model.Vehicle(env.opp_car, env.dt, origin_point)
        opp_controller = MPC_controller(opp_car_model)

        # ------------------- Grid Map ------------------------
        # Define the properties of the abstraction grid
        lane_width = 4
        cell_size = np.array([lane_width, lane_width]) # Size of each grid cell in meters
        grid_shape = (5, 8) # Grid dimensions (rows, columns)
        grid = Griding(env.world, origin_point, grid_shape, cell_size) # Visualization tool for the grid


        # ------------------- Labelling ------------------------
        # The labelling function can be extracted from the semantic map
        # This function updates the labels of grid cells dynamically based on the opponent's position
        def dyn_labelling(sta_labels, v_pos, v_action):
            labels = sta_labels.copy()
            g_x = int(np.floor(v_pos[0] / cell_size[0]))
            g_y = int(np.floor(v_pos[1] / cell_size[1]))
            # v_2 = ((g_x + min(0, 2 * v_action[0])) * cell_size[0], (g_x + max(1, 1 + 2 * v_action[0])) * cell_size[0],
            #        (g_y + min(0, 2 * v_action[1])) * cell_size[1], (g_y + max(1, 1 + 2 * v_action[1])) * cell_size[1])
            
            # Predict future unsafe area (v_1) based on opponent action
            v_1 = ((g_x + min(0, v_action[0])) * cell_size[0], (g_x + max(1, 1 + v_action[0])) * cell_size[0],
                   (g_y + min(0, v_action[1])) * cell_size[1], (g_y + max(1, 1 + v_action[1])) * cell_size[1])
            # Current area occupied by opponent (v_0)
            v_0 = (g_x * cell_size[0], (g_x + 1) * cell_size[0], g_y * cell_size[1], (g_y + 1) * cell_size[1])
            labels[v_0] = "v" # Mark current cell as Vehicle
            labels[v_1] = "f" # Mark next predicted cell as Future unsafe
            return labels

        # STATIC LABELS: Define fixed regions in the grid (Obstacles, Intersections, Targets)
        # Each key is a tuple (x_min, x_max, y_min, y_max) relative to the origin_point
        static_label = {(1, 4 * cell_size[0], 2 * cell_size[1], 7 * cell_size[1]): "c", # c: Collision/Intersection area
                      (4 * cell_size[0], 5 * cell_size[0],
                       5 * cell_size[1], 6 * cell_size[1]): "t", # t: Target area
                      (2 * cell_size[0], 5 * cell_size[0], 0, 5 * cell_size[1]): "o", # o: Obstacle (Bottom Right)
                      (3 * cell_size[0], 5 * cell_size[0], 6 * cell_size[1], 8 * cell_size[1]): "o", # o: Obstacle (Top Right)
                      (0, 1 * cell_size[0], 0, 8 * cell_size[1]): "o"} # o: Obstacle (Left Border)


        # ---------- Initialization for Ego's Decision Maker --------------------
        ego_local_state = (0, 0, 0)
        ego_prod_state = (0, 1, 1) # Initial product state (automata state)
        env_abs_state = 1
        oppo_abs_state = [-1, -1]
        _oppo_abs_state = [-1, -1]
        _ego_abs_state = [-1, -1]

        # Initialize the Abstraction Model
        region_size = (grid_shape[0] * cell_size[0], grid_shape[1] * cell_size[1])
        abs_model = Abstraction(region_size, cell_size, ego_local_state, static_label,scenario="intersection")
        
        # Define costs for different labels
        cost_map = {"c": 5, "o": 5, "v": 10, "f": 5}
        
        # Initialize Decision Maker components
        MDP_env = traffic_light_model()
        des_maker = Risk_LTL(abs_model, abs_model.MDP, MDP_env, safe_spec.dfa, scltl_spec.dfa, cost_map)
        
        target_abs_state_sys = None
        opt_action = None
        target_point = None

        EgoViewFlag = True
        OfflineCalFlag = False
        Green_light_time = 250
        des_speed = 1.5


        if EgoViewFlag:
            display = pygame.display.set_mode(
                (800, 600),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            clock = pygame.time.Clock()
            # time.sleep(15)

        stage = 0
        iter = 0
        planned_path = None
        decision_risk = 0

        # Setup plotting for risk monitoring
        figure = plt.figure(figsize=(10, 8))
        ax = figure.add_subplot()
        plt.pause(0.01)
        risk_history = []
        decision_risk_history = {}
        if OfflineCalFlag:
            decision_risk_history = np.load("./offline_policy/decision_risk_history.npy", allow_pickle=True).item()
            # recorded_risk_history = np.load("./offline_policy/risk_history_0.5.npy", allow_pickle=True)


        # Move spectator camera to a good view point
        spectator = env.world.get_spectator()
        spectator.set_transform(carla.Transform(carla.Location(x=70, y=50, z=35),
                                          carla.Rotation(pitch=-50)))

        while running:
            iter += 1

            env.world.tick()

            # Update vehicle physics models with latest data from simulation
            ego_car_model.update()
            opp_car_model.update()
            
            # Check if state has changed (e.g. moved to a new grid cell)
            # Only re-calculate decision if the discret state of ego or opponent changes
            change_flag = False
            if stage == 1:
                ego_local_state = ego_car_model.get_local_state()
                oppo_local_state = opp_car_model.get_local_state()
                ego_abs_state_index, ego_abs_state = abs_model.get_abs_ind_state(ego_local_state)
                oppo_abs_state = abs_model.get_abs_state(oppo_local_state)

                change_flag = ((target_abs_state_sys is None)
                                or ((ego_abs_state[0] != _ego_abs_state[0])
                                    or (ego_abs_state[1] != _ego_abs_state[1]))
                               or ((oppo_abs_state[0] != _oppo_abs_state[0])
                                   or (oppo_abs_state[1] != _oppo_abs_state[1])))

            # ----------- Decision Making Step -----------------------------
            if change_flag:
                _oppo_abs_state = oppo_abs_state
                _ego_abs_state = ego_abs_state
                joint_state = (ego_abs_state[0], ego_abs_state[1], oppo_abs_state[0], oppo_abs_state[1], env_abs_state)
                
                if OfflineCalFlag:
                    # Use pre-calculated policy if available
                    optimal_policy = get_offline_policy(joint_state)
                    decision_risk = decision_risk_history[joint_state]
                    if optimal_policy is not None:
                        planned_path = des_maker.get_opt_path(ego_prod_state, optimal_policy, ego_abs_state, env_abs_state)
                        decision_index, ego_prod_state = des_maker.offline_update(ego_prod_state, optimal_policy, ego_abs_state_index, env_abs_state)
                    else:
                        OfflineCalFlag = False
                        continue
                else:
                    # Online Decision Making
                    # 1. Update labels based on new opponent position
                    label_func = dyn_labelling(static_label, oppo_local_state, [0, -1])
                    abs_model.update(ego_local_state, label_func)
                    
                    # 2. Re-initialize MDP and Decision Maker with updated Environment
                    MDP_env = traffic_light_model()
                    des_maker = Risk_LTL(abs_model, abs_model.MDP, MDP_env, safe_spec.dfa, scltl_spec.dfa, cost_map)
                    decision_risk_history[joint_state] = decision_risk
                    
                    # 3. Solve for optimal policy using Risk-aware LTL
                    ego_prod_state, decision_index, optimal_policy, decision_risk = des_maker.update(ego_prod_state,
                                                                                         ego_abs_state_index,
                                                                                         env_abs_state, risk_th=0.1)
                    # 4. Get planned path for visualization
                    planned_path = des_maker.get_opt_path(ego_prod_state, optimal_policy, ego_abs_state, env_abs_state)
                    #policy_record(optimal_policy, joint_state)

                # Extract optimal action and target state
                opt_action = abs_model.action_set[decision_index]
                des_speed = np.hypot(opt_action[0], opt_action[1]) * 1.5
                target_abs_state_sys = ego_abs_state + opt_action

            if decision_risk is not None:
                risk_history.append(decision_risk * 2)
            ego_state = (ego_car_model.x, ego_car_model.y)

            print("ego_state", ego_state)
            print("stage", stage)
            
            # --- State Machine for Simulation Stages ---
            # Stage 0: Initial approach to start point
            if stage == 0:
                target_point = (88, 39, 0.3)
                if reachability_check(ego_state, target_point[:2], 1):
                    stage = 1
            # Stage 1: Autonomous Decision Making Active
            elif stage == 1:
                grid_center_x = (target_abs_state_sys[0] + 0.5) * cell_size[0] + origin_point.x
                grid_center_y = (target_abs_state_sys[1] + 0.5) * cell_size[1] + origin_point.y
                    
                carla_map = env.world.get_map()
                center_loc = carla.Location(x=grid_center_x, y=grid_center_y, z=origin_point.z)
                waypoint = carla_map.get_waypoint(center_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                target_point = (waypoint.transform.location.x, 
                                waypoint.transform.location.y, 
                                -90) 
                grid.draw_grid_map(time=0.5)
                grid.draw_path(planned_path)
                print("Path", planned_path)
                if reachability_check(ego_state, (99, 59.35), 2):
                    stage = 2
                    np.save("offline_policy/risk_history_0.5.npy", risk_history)
                    np.save("./offline_policy/decision_risk_history.npy", decision_risk_history)
                    decision_risk = 0
            # Stage 2: Goal reached, move away
            elif stage == 2:
                target_point = (130, 59.35, 0.3)

            # ----------- Control -----------------------------
            # Execute low-level control (MPC) to reach target point
            control_cmd = ego_controller.solve(target_point, des_speed)
            env.ego_car.apply_control(control_cmd)
            
            # ----------- Opponent Car Logic -----------------------------
            print("env_abs_state", env_abs_state)
            opp_car_model.update()
            oppo_des_speed = 1
            if iter >= Green_light_time:
                # Simple behavior: move forward if not at end
                if oppo_abs_state != [-1, -1]:
                    oppo_target_abs_state = [oppo_abs_state[0], oppo_abs_state[1] - 1]
                else:
                    oppo_target_abs_state = [2, 7]

                oppo_target_point = ((oppo_target_abs_state[0] + 1/2) * cell_size[0] + origin_point.x,
                                     (oppo_target_abs_state[1] + 1/2) * cell_size[1] + origin_point.y,
                                     -90)
                oppo_control_cmd = opp_controller.solve(oppo_target_point, oppo_des_speed)
                env.opp_car.apply_control(oppo_control_cmd)

                # Simulated traffic light sensor
                if env.is_green_light(iter, Green_light_time):
                    env_abs_state = 0
                else:
                    env_abs_state = 1



            # ----------- Print Risk -----------------------------
            risk_plot(ax, risk_history, soft_th=0.4, hard_th=0.8)
            # ----------- Traffic Light -----------------------------

            if EgoViewFlag:
                clock.tick_busy_loop(20)
                display.blit(env.camera_surface, (0, 0))
                pygame.display.flip()

    finally:
        env.__del__()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Exit by user')


