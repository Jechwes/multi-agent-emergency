"""
roundabout.py
=============
CARLA environment setup for the roundabout scenario.

Handles world initialisation (Town03_Opt, synchronous mode), ego vehicle
spawning, and pedestrian (walker) management.  The pedestrian patrols
radially between an inner and outer radius, reversing direction at each
boundary.

``Environment`` is instantiated once at startup and destroyed in the
``finally`` block; CARLA world settings are restored on destruction.
"""
import carla
import random
import math
import numpy as np
from queue import Queue
import weakref

class Environment:
    def __init__(self, args, ego_transform=None):
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(3.0)
        self.world = self.client.load_world('Town03_Opt')

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

        # -------------- Pedestrian (walker) settings -------------- #
        self.pedestrian = None
        self.ped_controller = None

    # Function to add cars
    def add_car(self, spawn_point, color):
        model3_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        model3_bp.set_attribute('color', color)
        car = self.world.spawn_actor(model3_bp, spawn_point)
        return car

    # ------------------------------------------------------------------
    # Pedestrian (walker) spawning
    # ------------------------------------------------------------------

    def spawn_pedestrian(
        self,
        spawn_location: carla.Location,
        direction: carla.Vector3D = None,
        speed: float = 1.4,
    ):
        """
        Spawn a CARLA walker at *spawn_location* and start it walking in
        *direction* at *speed* [m/s].  The walker + controller are stored
        on ``self.pedestrian`` / ``self.ped_controller``.

        Parameters
        ----------
        spawn_location : carla.Location
            World position for the pedestrian.
        direction      : carla.Vector3D, optional
            Initial walking direction (unit vector).  Defaults to (1, 0, 0).
        speed          : float
            Walking speed [m/s]  (default 1.4 ≈ normal walk).
        """
        bp_lib = self.world.get_blueprint_library()
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        walker_bp = random.choice(walker_bps)

        # Make the walker invincible so it doesn't ragdoll on collision
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'true')

        spawn_tf = carla.Transform(spawn_location)
        self.pedestrian = self.world.spawn_actor(walker_bp, spawn_tf)

        # Attach an AI controller (required for apply_control)
        ctrl_bp = bp_lib.find('controller.ai.walker')
        self.ped_controller = self.world.spawn_actor(
            ctrl_bp, carla.Transform(), attach_to=self.pedestrian,
        )

        # Initial direction
        if direction is None:
            direction = carla.Vector3D(x=1.0, y=0.0, z=0.0)

        self._ped_speed = speed
        self._ped_direction = direction

        # Apply initial walk command
        control = carla.WalkerControl(
            direction=self._ped_direction,
            speed=self._ped_speed,
        )
        self.pedestrian.apply_control(control)

        print(f"[Env] Pedestrian spawned at "
              f"({spawn_location.x:.1f}, {spawn_location.y:.1f})")

    def update_pedestrian_patrol(
        self,
        centre: tuple,
        inner_radius: float,
        outer_radius: float,
        section_angle: float,
        wait_at_outer: float = 0.0,
    ) -> None:
        """
        Make the pedestrian patrol radially between *inner_radius* and
        *outer_radius* at a fixed angular position.

        Call this every simulation tick.  When the pedestrian crosses a
        boundary radius it reverses direction.

        Parameters
        ----------
        wait_at_outer : float
            Seconds to wait at the outer edge before walking back inward.
        """
        if self.pedestrian is None:
            return

        loc = self.pedestrian.get_location()
        dx = loc.x - centre[0]
        dy = loc.y - centre[1]
        r = math.sqrt(dx * dx + dy * dy)

        # Radial unit vector (outward)
        if r < 1e-3:
            return
        ur_x = dx / r
        ur_y = dy / r

        # Reverse direction at boundaries
        # _ped_outward: True = walking outward, False = walking inward
        if not hasattr(self, '_ped_outward'):
            self._ped_outward = False          # start walking inward
            self._ped_wait_ticks = 0           # tick counter for pause
            self._ped_wait_duration = int(wait_at_outer / self.dt) if self.dt > 0 else 0

        if r >= outer_radius:
            if self._ped_outward:               # just arrived at outer edge
                self._ped_wait_ticks = self._ped_wait_duration
            self._ped_outward = False           # turn inward
        elif r <= inner_radius:
            self._ped_outward = True            # turn outward

        # If waiting at a boundary, stay still and count down
        if self._ped_wait_ticks > 0:
            self._ped_wait_ticks -= 1
            control = carla.WalkerControl(
                direction=carla.Vector3D(0, 0, 0),
                speed=0.0,
            )
            self.pedestrian.apply_control(control)
            return

        sign = 1.0 if self._ped_outward else -1.0
        direction = carla.Vector3D(
            x=float(sign * ur_x),
            y=float(sign * ur_y),
            z=0.0,
        )

        control = carla.WalkerControl(
            direction=direction,
            speed=self._ped_speed,
        )
        self.pedestrian.apply_control(control)

    def __del__(self):
        # Destroy pedestrian + controller first
        if getattr(self, 'ped_controller', None) is not None:
            try:
                if getattr(self.ped_controller, 'is_alive', True):
                    self.ped_controller.stop()
                self.ped_controller.destroy()
            except RuntimeError:
                pass
        if getattr(self, 'pedestrian', None) is not None:
            try:
                self.pedestrian.destroy()
            except RuntimeError:
                pass
        if getattr(self, 'world', None) is not None and getattr(self, 'original_settings', None) is not None:
            self.world.apply_settings(self.original_settings)
        print('done')

