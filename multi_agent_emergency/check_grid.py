import argparse
import time
import os
import sys
try:
    import numpy as np
except ImportError:
    raise RuntimeError('import error!')

from queue import Empty
from scenarios.roundabout import *
import control.vehicle_model as model
from control.trackingMPC import MPC_controller
from perception.grid_polar import Griding
import math

# Allow imports from siblings when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abstraction.frenet_lanelet import Lanelet, LaneletMap

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

        # -------------- Polar grid (existing) -----------------
        origin = carla.Location(x=-0.5, y=0.5, z=0.2)
        # specific_angle_step e.g. 10 degrees
        grid = Griding(env.world, origin, lane_start=13.5, lane_amount=4, cell_width=4, angle_step=12)
        grid_time = 30
        grid.draw_grid_map_polar(grid_time)

        # Highlight a specific cell at [ring_idx, sector_idx, num_rings, num_sectors]
        grid.draw_box([3, 16, 1, 1], grid_time)

        # -------------- Frenet lanelet overlay -----------------
        # Parameters must match the polar grid above
        org_x      = origin.x          # -0.5
        org_y      = origin.y          #  0.5
        lane_start = 13.5              # inner radius of ring 0 [m]
        cell_width = 4.0               # radial width per ring  [m]
        lane_amount = 4                # number of rings
        n_circ     = 120               # waypoints per full circle (finer = smoother spline)

        # Build one circular Lanelet per ring.
        # The centre-line of ring i sits at radius = lane_start + (i + 0.5) * cell_width.
        angles = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
        waypoint_lists = []
        ring_radii = []
        for i in range(lane_amount):
            r = lane_start + (i + 0.5) * cell_width   # ring centre radius
            ring_radii.append(r)
            wps = np.column_stack([
                org_x + r * np.cos(angles),
                org_y + r * np.sin(angles),
            ])
            # Close the loop so the spline wraps smoothly
            wps = np.vstack([wps, wps[0]])
            waypoint_lists.append(wps)

        # chain_s=False: each ring is an independent lanelet (parallel, not sequential)
        lmap = LaneletMap.from_waypoint_lists(
            waypoint_lists,
            lane_widths=[cell_width] * lane_amount,
            chain_s=False,
        )

        # Draw: centre-lines + lane boundaries + transverse s-grid lines.
        # s_grid_step matches your Frenet MDPModel grid resolution (here same as cell_width).
        lmap.draw_in_carla(
            env.world,
            z=origin.z + 0.15,       # draw slightly above polar grid
            n_samples=300,
            s_grid_step=cell_width,  # transverse line every 4 m along arc
            life_time=grid_time,
        )

        print(f"Frenet LaneletMap drawn: {lmap}")
        print(f"  Ring radii: {ring_radii}")
        for lane in lmap.lanelets:
            lb, ub = lane.s_bounds()
            print(f"  Lane {lane.lane_id}: circumference = {lane.length:.1f} m"
                  f",  s ∈ [{lb:.1f}, {ub:.1f}]"
                  f",  κ_max = {max(abs(lane.curvature(s)) for s in np.linspace(0, lane.length, 50)):.4f} 1/m")


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

