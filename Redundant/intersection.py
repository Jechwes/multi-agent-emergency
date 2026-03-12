import carla
import random
import pygame
# import scenarios
import numpy as np
from queue import Queue
import weakref




class Environment:

    def __init__(self, args):
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(3.0)
        self.world = self.client.load_world('Town01')

        # --------- Weather Setting ----------------------
        # weather = carla.WeatherParameters(
        #     cloudiness=35.0,
        #     precipitation=0.0,
        #     sun_altitude_angle=60.0)
        # self.world.set_weather(weather)

        # buildings = self.world.get_environment_objects(carla.CityObjectLabel.Buildings)
        # objects_to_toggle = [building.id for building in buildings]
        # self.world.enable_environment_objects(objects_to_toggle, False)

        self.original_settings = self.world.get_settings()
        self.dt = 0.03
        random.seed(3)

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = self.dt
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.sensor_queue = Queue()

        # --------- Ego Vehicle Setting ----------------------
        # _init_loc = carla.Location(x=88, y=38, z=0.3)
        _init_loc = carla.Location(x=88, y=18, z=0.3)
        _init_rot = carla.Rotation(pitch=0, yaw=90, roll=0)
        ego_init_tran = carla.Transform(_init_loc, _init_rot)
        self.ego_car = self.add_car(ego_init_tran, "50, 50, 200")

        # --------- Opponent Vehicle Setting ----------------------
        _init_loc = carla.Location(x=92, y=73, z=0.3)
        _init_rot = carla.Rotation(pitch=0, yaw=-90, roll=0)
        opp_init_tran = carla.Transform(_init_loc, _init_rot)
        self.opp_car = self.add_car(opp_init_tran, "200, 20, 20")

        self.zombie_cars = []

        #---------------- Traffic Light ----------------------
        self.traffic_lights = self.world.get_actors().filter('traffic.traffic_light')

        self.camera_surface = None
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera = self.world.spawn_actor(camera_bp,
                                   carla.Transform(carla.Location(x=-5.5, z=2.5),
                                                   carla.Rotation(pitch=8.0)),
                                   self.ego_car,
                                   carla.AttachmentType.SpringArm)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: Environment.draw_image(weak_self, image))

    def add_car(self, spawn_point, color):
        model3_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        model3_bp.set_attribute('color', color)
        car = self.world.spawn_actor(model3_bp, spawn_point)
        car.set_autopilot(False)
        return car

    @staticmethod
    def draw_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        # if self.recording:
        #     image.save_to_disk('_out/%08d' % image.frame)

    def is_green_light(self, time, Green_light_time):
        wp = self.world.get_map().get_waypoint(self.ego_car.get_location())
        traffic_light_list = self.world.get_traffic_lights_from_waypoint(wp, 20)
        if len(traffic_light_list) != 0:
            if time >= Green_light_time:
                for traffic_light in self.traffic_lights:
                    # traffic_light.is_frozen = True
                    traffic_light.set_state(carla.TrafficLightState.Green)
            if traffic_light_list[0].get_state() == carla.TrafficLightState.Green:
                return True
            else:
                return False
        else:
            return True

    def __del__(self):
        self.world.apply_settings(self.original_settings)
        print('done')