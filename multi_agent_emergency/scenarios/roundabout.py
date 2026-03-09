import carla
import random
import numpy as np
from queue import Queue
import weakref

class Environment:
    def __init__(self, args, ego_transform=None):
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(3.0)
        self.world = self.client.load_world('Town03')

        self.original_settings = self.world.get_settings()
        self.dt = 0.01
        random.seed(3)

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = self.dt
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.sensor_queue = Queue()

        # -------------- Vehicle settings -------------- #
        if ego_transform is None:
            # Legacy default (section ≈ 3, lane 1 of Town03 roundabout)
            _init_loc = carla.Location(x=-2.1, y=20.2, z=0.3)
            _init_rot = carla.Rotation(pitch=0.0, yaw=0, roll=0.0)
            ego_transform = carla.Transform(_init_loc, _init_rot)

        self.ego_car = self.add_car(ego_transform, "50, 50, 200")

        # # Vehicle 2
        # _init_loc = carla.Location(x=-11.1, y=16.5, z=0.3)
        # _init_rot = carla.Rotation(pitch=0.0,yaw=35,roll=0.0)
        # car_init_tran = carla.Transform(_init_loc, _init_rot)
        # self.ego_car2 = self.add_car(car_init_tran, "100, 0, 200")

        # # Vehicle 3
        # _init_loc = carla.Location(x=-19.2, y=6, z=0.3)
        # _init_rot = carla.Rotation(pitch=0.0,yaw=67.6,roll=0.0)
        # car_init_tran = carla.Transform(_init_loc, _init_rot)
        # self.ego_car3 = self.add_car(car_init_tran, "0, 200, 0")

        # Vehicle 4
        # _init_loc = carla.Location(x=-17.6, y=-9, z=0.3)
        # _init_rot = carla.Rotation(pitch=0.0,yaw=115,roll=0.0)
        # car_init_tran = carla.Transform(_init_loc, _init_rot)
        # self.ego_car4 = self.add_car(car_init_tran, "200, 50, 50")

        self.zombie_cars = []

    # Function to add cars
    def add_car(self, spawn_point, color):
        model3_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        model3_bp.set_attribute('color', color)
        car = self.world.spawn_actor(model3_bp, spawn_point)

        # PORT BUSY? WTF HAPPENS HERE?
        # try:
        #     car.set_autopilot(True) 
        # except RuntimeError:
        #     # Fallback or just print
        #     print("Failed to set autopilot, TM port might be busy.")
         
        return car

    def __del__(self):
        self.world.apply_settings(self.original_settings)
        print('done')

