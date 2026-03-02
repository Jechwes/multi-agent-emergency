import argparse
import time
try:
    import numpy as np # Fixed typo 'as py'
    import sys

    from os import path as osp
except ImportError:
    raise RuntimeError('import error!')

from queue import Empty
from scenarios.roundabout import *
import control.vehicle_model as model
from control.trackingMPC import MPC_controller 
from perception.grid_polar import Griding
import math

def main():
    argparser = argparse.ArgumentParser(description='Carla ArgParser practice')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    args = argparser.parse_args()
    running = True
    env = None

    

    try:
        env = Environment(args)
        env.world.tick()

        # Initialize spectator 
        spectator = env.world.get_spectator()
        transform = spectator.get_transform()
        spectator.set_transform(carla.Transform(carla.Location(z=50), carla.Rotation(pitch=-90))) 

        # -------------- Grid -----------------
        origin = carla.Location(x=-0.5, y=0.5, z=0.2)
        # specific_angle_step e.g. 10 degrees
        grid = Griding(env.world, origin, lane_start=13.5, lane_amount=4, cell_width=4, angle_step=12)
        grid_time = 30
        grid.draw_grid_map_polar(grid_time)

        # Highlight a specific cell at [ring_idx, sector_idx, num_rings, num_sectors]
        grid.draw_box([3, 16, 1, 1], grid_time) 


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

