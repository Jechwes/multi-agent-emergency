import math
import casadi as ca
import carla
import numpy as np
import bisect

class MPC_controller:
    def __init__(self, car_model):

        self.dt = 0.1 # time step
        self.horizon = 5
        self.WB = car_model.wheelbase
        self.car = car_model

        # MPC config
        self.opti = ca.Opti()
        self.x_dim = 4
        self.u_dim = 2
        self.X = None
        self.U = None
        self.X_ref = self.opti.parameter(self.x_dim, self.horizon)
        self.X_0 = self.opti.parameter(self.x_dim)

        self.Q = np.diag([5.0, 5.0, 1.0, .0])  # penalty for states
        self.Qf = np.diag([2.0, 2.0, 3.0, .0])  # penalty for end state
        self.R = np.diag([1, 1]) # penalty for inputs

        self.v_bound = np.array([-1, 1]) * car_model.speed_max
        self.acc_bound = np.array([-1, 1]) * car_model.acc_max
        self.delta_bound = np.array([-1, 1]) * car_model.steer_max

        self.Acc_Table = {0: 0, 0.2: 0.5, 0.4: 0.8, 0.6: 0.9, 0.8: 0.95, 1: 1}
        self.prob_init()

    def prob_init(self):
        # Define the state and control variables
        x = ca.SX.sym('x')  # x position
        y = ca.SX.sym('y')  # y position
        v = ca.SX.sym('v')  # velocity
        yaw = ca.SX.sym('yaw')  # steering angle (rad)
        a = ca.SX.sym('a')  # velocity
        theta = ca.SX.sym('theta')  # orientation angle (rad)
        states = ca.vertcat(x, y, v, yaw)
        controls = ca.vertcat(a, theta)

        # Bicycle abstraction dynamics
        xdot = v * ca.cos(yaw)
        ydot = v * ca.sin(yaw)
        vdot = a
        yawdot = v / self.WB * ca.tan(theta)

        # Control horizon
        state_dot = ca.vertcat(xdot, ydot, vdot, yawdot)

        # Function to integrate dynamics over each interval
        integrate_f = ca.Function('integrate_f', [states, controls], [state_dot])

        # Objective function and constraints
        self.X = self.opti.variable(self.x_dim, self.horizon + 1)  # state trajectory
        self.U = self.opti.variable(self.u_dim, self.horizon)  # control trajectory

        cost = 0
        # Setup cost function and constraints
        for k in range(0, self.horizon):
            # Cost function (minimize distance to reference point)
            state_error = self.X[:, k] - self.X_ref[:, k]
            state_error[3] = ca.atan2(ca.sin(state_error[3]), ca.cos(state_error[3]))
            cost += ca.mtimes([state_error.T, self.Q, state_error]) + ca.mtimes(
                [self.U[:, k].T, self.R, self.U[:, k]])

            # System dynamics as constraints
            st_next = self.X[:, k] + integrate_f(self.X[:, k], self.U[:, k]) * self.dt
            self.opti.subject_to(self.X[:, k + 1] == st_next)
            self.opti.subject_to(self.U[0, k] > self.acc_bound[0])
            self.opti.subject_to(self.U[0, k] < self.acc_bound[1])
            self.opti.subject_to(self.U[1, k] > self.delta_bound[0])
            self.opti.subject_to(self.U[1, k] < self.delta_bound[1])
        # Boundary conditions
        self.opti.subject_to(self.X[:, 0] == self.X_0)  # initial condition
        # Solver configuration
        self.opti.minimize(cost)
        # TO ENABLE SOLVER OUTPUT: change print_level to 5 and print_time to True
        opts = {'ipopt': {'print_level': 0}, 'print_time': False}
        self.opti.solver('ipopt', opts)

    def solve(self, target, des_speed):
        self.car.update()
        X_0 = np.array([self.car.x, self.car.y, self.car.v, self.car.yaw])
        self.opti.set_value(self.X_0, X_0)
        ref_traj = self.gen_ref_traj(target, des_speed)
        self.opti.set_value(self.X_ref, ref_traj)
        sol = self.opti.solve()

        opt_states = sol.value(self.X)
        opt_inputs = sol.value(self.U)
        # TO ENABLE SOLVER OUTPUT: uncomment the line below
        # print("opt_inputs", opt_inputs[:, 1])
        return self.gen_cmd(opt_inputs[0, 1], opt_inputs[1, 1])

    def gen_cmd(self, acc_cmd, delta_cmd):
        cmd = carla.VehicleControl()
        cmd.steer = max(min(delta_cmd/self.delta_bound[1], 1), -1)
        index = bisect.bisect(list(self.Acc_Table.keys()), abs(acc_cmd)/self.acc_bound[1])-1

        if acc_cmd < 0:
            cmd.throttle = 0
            cmd.brake = abs(acc_cmd)/self.acc_bound[1]
        else:
            cmd.throttle = list(self.Acc_Table.values())[index]
            cmd.brake = 0
        return cmd

    def gen_ref_traj(self, target, des_speed):
        z_ref = np.zeros((self.x_dim, self.horizon))

        for i in range(0, self.horizon):

            z_ref[0, i] = target[0]
            z_ref[1, i] = target[1]
            z_ref[2, i] = des_speed
            z_ref[3, i] = target[2]

        return z_ref

    def solve_trajectory(self, ref_traj_4xH):
        """
        Solve MPC with a pre-built (4, horizon) reference trajectory.

        Parameters
        ----------
        ref_traj_4xH : np.ndarray (4, horizon)
            Each column is [x, y, v, yaw] for one horizon step.

        Returns
        -------
        carla.VehicleControl
        """
        self.car.update()
        X_0 = np.array([self.car.x, self.car.y, self.car.v, self.car.yaw])
        self.opti.set_value(self.X_0, X_0)
        self.opti.set_value(self.X_ref, ref_traj_4xH)
        sol = self.opti.solve()
        opt_inputs = sol.value(self.U)
        return self.gen_cmd(opt_inputs[0, 0], opt_inputs[1, 0])









