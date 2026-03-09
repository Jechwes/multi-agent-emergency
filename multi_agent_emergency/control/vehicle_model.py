import math
import numpy as np

class Vehicle:
    def __init__(self, vehicle, dt, origin_point):
        self.vehicle = vehicle
        self.target_vel = 10
        self.dt = dt
        self.steer_max = math.pi / 5
        self.speed_max = 40
        self.acc_max = 8
        self.steer_max = np.deg2rad(35.0)
        self.steer_change_max = np.deg2rad(15.0)  # maximum steering speed [rad/s]
        wb_vec = vehicle.get_physics_control().wheels[0].position - vehicle.get_physics_control().wheels[2].position
        self.wheelbase = np.sqrt(wb_vec.x**2 + wb_vec.y**2 + wb_vec.z**2)/100
        self.merge_length = 0
        self.shape = [self.vehicle.bounding_box.extent.x, self.vehicle.bounding_box.extent.y,
                      self.vehicle.bounding_box.extent.z]
        self.origin_point = origin_point
        self.x = None
        self.y = None
        self.v = None
        self.acc = None
        self.yaw = None
        self.steer = None
        self.update()  # initialize

    def update(self):
        self.x = self.vehicle.get_location().x
        self.y = self.vehicle.get_location().y

        _v = self.vehicle.get_velocity()
        _acc = self.vehicle.get_acceleration()
        self.v = np.sqrt(_v.x ** 2 + _v.y ** 2)
        self.acc = np.sqrt(_acc.x ** 2 + _acc.y ** 2)
        self.steer = self.vehicle.get_control().steer
        self.merge_length = max(4 * self.v, 12)

        fwd = self.vehicle.get_transform().get_forward_vector()
        self.yaw = math.atan2(fwd.y, fwd.x)

    def predict(self, a, delta):
        '''Bicycle Model for vehicles'''
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        self.yaw += self.v / self.wheelbase * math.tan(delta) * self.dt
        self.v += a * self.dt

    def get_local_state(self):
        return np.array([self.x - self.origin_point.x + np.cos(self.yaw) * self.wheelbase/2,
                         self.y - self.origin_point.y + np.sin(self.yaw) * self.wheelbase/2,
                         self.yaw])