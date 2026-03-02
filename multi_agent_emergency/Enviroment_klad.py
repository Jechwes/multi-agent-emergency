import carla
import random
import time

## Connect to the client
client = carla.Client('localhost', 2000)
client.set_timeout(10.0) # Always good to set a timeout in case the server is slow

## 1. LOAD THE MAP FIRST
# This reloads the simulation. It returns the new world object directly.
print("Loading Map...")
world = client.load_world('Town03') 

## Spectator navigation
spectator = world.get_spectator()
transform = spectator.get_transform()
spectator.set_transform(carla.Transform(carla.Location(z=50), carla.Rotation(pitch=-90))) 
# I set the camera high up looking down so you can actually see the cars spawning

## Adding NPCs
vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
spawn_points = world.get_map().get_spawn_points()

print(f"Found {len(spawn_points)} spawn points.")

spawn_points = world.get_map().get_spawn_points()
random.shuffle(spawn_points)  # <--- Shuffle the list once

print(f"Found {len(spawn_points)} spawn points.")

# Spawn NPCs
# for i in range(0, 50):
#     if not spawn_points:
#         print("Ran out of spawn points!")
#         break
#         
#     point = spawn_points.pop() # <--- Take a unique point out of the list
#     bp = random.choice(vehicle_blueprints)
#     
#     # We can safely use spawn_actor now because we know the point is unique from our list
#     # But try_spawn_actor is still good practice in case of debris/physics lag
#     npc = world.try_spawn_actor(bp, point)
#     
#     if npc is not None:
#         npc.set_autopilot(True)
# 
# print("NPCs spawned.")
# 
# # Ego Vehicle
# if spawn_points:
#     ego_bp = random.choice(vehicle_blueprints)
#     ego_point = spawn_points.pop() # <--- Take the NEXT unique point
#     
#     # Use try_spawn_actor to be safe, or handle the exception
#     ego_vehicle = world.try_spawn_actor(ego_bp, ego_point)
#     
#     if ego_vehicle is not None:
#         ego_vehicle.set_autopilot(True)
#         print("Ego vehicle spawned!")
#     else:
#         print("Ego vehicle spawn failed (collision with debris/non-list object).")
# else:
#     print("No spawn points left for Ego vehicle!")

## Add sensors
# camera_init_trans = carla.Transform(carla.Location(z=1.5, x=2.5)) # Slightly forward so you see the road
# camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
# camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# Start camera callback
# NOTE: Ensure the 'out' folder exists in your directory, or this will fail
#camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))

print("Simulation running. Press Ctrl+C to stop.")

## Keep the script running
try:
    while True:
        world.wait_for_tick()

        # car_loc = ego_vehicle.get_location()
        spec_trans = spectator.get_transform()
        spec_loc = spec_trans.location
        spec_rot = spec_trans.rotation

        print(f"SPEC Location: [{spec_loc.x:6.1f}, {spec_loc.y:6.1f}, {spec_loc.z:6.1f}]  |  "
              f"SPEC Rotation: [P:{spec_rot.pitch:6.1f}, Y:{spec_rot.yaw:6.1f}, R:{spec_rot.roll:6.1f}]", end='\r')
        
except KeyboardInterrupt:
    print("Stopping...")
    # Clean up sensors (Good practice to destroy sensors before exiting)
    # camera.stop()
    # camera.destroy()
    # ego_vehicle.destroy()
    print("Done.")