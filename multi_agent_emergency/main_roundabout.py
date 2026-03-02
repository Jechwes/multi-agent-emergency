import argparse
import time
try:
    import numpy as np # Fixed typo 'as py'
    import sys
    import os

    from os import path as osp
except ImportError:
    raise RuntimeError('import error!')

# Add the directory containing your local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
local_module_dir = os.path.join(script_dir, 'decision/')
for root, dirs, files in os.walk(local_module_dir):
    if root not in sys.path:
        sys.path.append(root)

from queue import Empty
from decision.specification.ltl_spec import Translate
from scenarios.roundabout import *
import control.vehicle_model as model
from control.trackingMPC import MPC_controller 
from perception.grid_polar import Griding
from decision.abstraction.abstract_polar import Abstraction
from decision.maker_roundabout import Risk_LTL
import math

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
    env = None

    # ---------- Specification Define --------------------
    # Define Linear Temporal Logic (LTL) specifications for safety and task completion.
    safe_spec = Translate("G(~n)", AP_set=['n'])

    # scltl_spec: "Finally (F) reach Target (t)"
    scltl_spec = Translate("F(t)", AP_set=['t'])

    try:
        env = Environment(args)
        env.world.tick()

        # Initialize spectator 
        spectator = env.world.get_spectator()
        transform = spectator.get_transform()
        spectator.set_transform(carla.Transform(carla.Location(z=50), carla.Rotation(pitch=-90)))

        # Define the origin point for the abstract grid system
        origin = carla.Location(x=-0.5, y=0.5, z=0.2)

        # Initialize vehicle physics model and MPC controller
        ego_car_model = model.Vehicle(env.ego_car, env.dt, origin)
        ego_controller = MPC_controller(ego_car_model) 

        # -------------- Grid -----------------
        # Settings
        lane_start = 13.5 #[17.5 is start of inner lane]
        lane_amount = 4
        angle_step = 12
        cell_width = 4

        # specific_angle_step e.g. 10 degrees
        grid = Griding(env.world, origin, lane_start, lane_amount, cell_width, angle_step)
        grid_time = 30
        grid.draw_grid_map_polar(grid_time)

        # --------------- Control --------------
        # Initialize controller
        car_model = model.Vehicle(env.ego_car, env.dt, origin)
        ego_controller = MPC_controller(car_model) 

        # ------------------- Labelling ------------------------
        # TODO: DYNAMIC LABELLING LATER TOEVOEGEN!

        # STATIC LABELS: Define fixed regions in the polar grid
        # Key: (ring_idx, sector_idx), Value: label
        static_label = {}
        n_sectors = int(360 / angle_step)
        
        for s in range(n_sectors):
            # Ring 0: Inner non-drivable
            static_label[(0, s)] = 'n'
            
            # Rings 1 & 2: Roundabout
            static_label[(1, s)] = 'r'
            static_label[(2, s)] = 'r'
            
            # Ring 3: Outer non-drivable (except sector 16)
            if s != 16:
                static_label[(3, s)] = 'n'
            else:
                static_label[(3, s)] = 't'

        # ---------- Initialization for Ego's Decision Maker --------------------
        ego_local_state = (0, 0, 0) 
        ego_prod_state = (0, 1, 1)  
        _ego_abs_state = [-1, -1]

        # Initialize the Abstraction Model (Polar Version)
        # origin, lane_start, lane_amount, cell_width, angle_step, initial_position, label_function, scenario
        abs_model = Abstraction(origin, lane_start, lane_amount, cell_width, angle_step, ego_local_state, static_label
        )
        
        # Define costs for different labels (Adjust keys to match your labels: n, r, t)

        # 'o': obstacle, 'n': non-drivable, 'r': road
        cost_map = {"n": 50, "r": 0, "t": 0}
        des_maker = Risk_LTL(abs_model, abs_model.MDP, safe_spec.dfa, scltl_spec.dfa, cost_map)
        
        stage = 0
        iter = 0
        planned_path = None
        target_abs_state_sys = None
        decision_index = None
        optimal_policy = None
        
        # Initial control target (start point)
        opt_action = (0, 0)
        des_speed = 0
        
        while running:
            iter += 1
            env.world.tick()

            # Update ego vehicle model
            ego_car_model.update()
            
            # Get current states
            ego_local_state = ego_car_model.get_local_state()
            ego_abs_state_index, ego_abs_state = abs_model.get_abs_ind_state(ego_local_state)
            
            # --- Check for State Change ---
            # Only update decision if we moved to a new cell or don't have a plan yet
            change_flag = False
            if (target_abs_state_sys is None) or \
               (ego_abs_state[0] != _ego_abs_state[0]) or \
               (ego_abs_state[1] != _ego_abs_state[1]):
                change_flag = True
                
            # --- Decision Making Step ---
            if change_flag:
                _ego_abs_state = ego_abs_state
                
                # TODO: Add dynamic labeling 
                abs_model.update(ego_local_state, static_label)

                if ego_abs_state_index is None:
                    # Vehicle out of bounds or invalid position
                    pass
                else: 
                    ego_prod_state, decision_index, optimal_policy, decision_risk = des_maker.update(
                        ego_prod_state,
                        ego_abs_state_index,
                        risk_th=0.1
                    )
                
                # Get abstract action
                if decision_index is not None:
                    opt_action = abs_model.action_set[decision_index]
                    
                    target_r = ego_abs_state[0] + opt_action[0]
                    target_theta = (ego_abs_state[1] + opt_action[1]) % n_sectors
                    target_abs_state_sys = [target_r, target_theta]
                    
                    des_speed = np.hypot(opt_action[0], opt_action[1]) * 10.0 # Speed scale factor
                        
                    planned_path = des_maker.get_opt_path(ego_prod_state, optimal_policy, ego_abs_state)
                
            ego_state = (ego_car_model.x, ego_car_model.y)    
            # --- Control Step ---
            if target_abs_state_sys is not None:
                if stage == 0:

                    t_r = lane_start + target_abs_state_sys[0] * cell_width + (cell_width/2)
                    t_theta_deg = target_abs_state_sys[1] * angle_step + (angle_step/2)
                    t_theta_rad = math.radians(t_theta_deg)
                
                    t_x = origin.x + t_r * math.cos(t_theta_rad)
                    t_y = origin.y + t_r * math.sin(t_theta_rad)
                
                    carla_map = env.world.get_map()
                    approx_loc = carla.Location(x=t_x, y=t_y, z=origin.z)
                
                    waypoint = carla_map.get_waypoint(approx_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                
                    if waypoint:
                        target_point = (waypoint.transform.location.x, 
                                        waypoint.transform.location.y, 
                                        waypoint.transform.rotation.yaw)
                    else:
                        t_yaw = t_theta_deg + 90 
                        target_point = (t_x, t_y, t_yaw)
                    
                    if reachability_check(ego_state, (-25.9, -7.6), 2):
                        stage = 1
                else:
                    target_point = (-65.1, -3.1, 0.3)
                    if reachability_check(ego_state, (-65.1, -3.1), 2):
                        print('Done!')
                        return

                
                # Visualize
                if planned_path:
                    grid.draw_path(planned_path)
                
                # MPC Control
                try:
                    control_cmd = ego_controller.solve(target_point, des_speed)
                    env.ego_car.apply_control(control_cmd)
                except Exception as e:
                    print(f"MPC Error: {e}")
                    env.ego_car.apply_control(carla.VehicleControl(brake=1.0))
            
    finally:
        if env is not None:
            env.__del__()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Exit by user')


# -------------------------------------------- OLD CODE
        

        # # 2. Get the Map for waypoint generation
        # carla_map = env.world.get_map()

        # spectator = env.world.get_spectator()
        # spectator.set_transform(carla.Transform(carla.Location(x=0, y=0, z=50),carla.Rotation(pitch=-90)))

        # tic = 0
        # target_speed = 20.0 # km/h or m/s depending on your controller logic
        
        # while running:
        #     tic += 1
        #     env.world.tick()
            
        #     # 3. Get Current Vehicle State
        #     car_model.update() # CRITICAL: Update the vehicle model with current state for MPC
        #     vehicle_transform = env.ego_car.get_transform()
            
        #     # 4. Find the Next Waypoint
        #     # We need to look far enough ahead for the MPC to have a stable target
        #     current_waypoint = env.world.get_map().get_waypoint(vehicle_transform.location, project_to_road=True)
            
        #     # Simple Logic: Follow the lane. 
        #     # If we are on the roundabout, next(dist) usually gives the next point on circle.
        #     next_waypoints = current_waypoint.next(7.0) # Look 5m ahead

        #     if next_waypoints:
        #          target_wp = next_waypoints[0]
        #          target_loc = target_wp.transform.location
                 
        #          # Prepare state for MPC: (x, y, yaw in degrees)
        #          # Note: CARLA yaw is in degrees, MPC might expect radians depending on implementation.
        #          # Usually this specific MPC implementation expects degrees based on main_intersection.py
        #          target_state = (target_loc.x, target_loc.y, target_wp.transform.rotation.yaw)
                 
        #          try:
        #              # Calculate control
        #              control_cmd = ego_controller.solve(target_state, target_speed)
                     
        #              # Apply Control
        #              env.ego_car.apply_control(control_cmd)
        #          except Exception as e:
        #              print(f"Controller Failed: {e}")
        #              # Emergency brake if controller determines infeasiblity
        #              env.ego_car.apply_control(carla.VehicleControl(brake=1.0))

        #     else:
        #          print("End of leg!")
        #          env.ego_car.apply_control(carla.VehicleControl(brake=1.0))
            
            # time.sleep(env.dt)

