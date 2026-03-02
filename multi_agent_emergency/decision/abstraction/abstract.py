#!/usr/bin/env python
import gurobipy as grb
import numpy as np
from abstraction.MDP import MDP
from scipy.stats import norm

class Abstraction:

    def __init__(self, map_range, map_res, initial_position, label_function, scenario):
        self.map_range = map_range
        self.map_res = map_res
        self.map_shape = None
        self.scenario = scenario
        self.state_set = self.gen_abs_state(map_range, map_res)
        self.action_set = self.gen_abs_action()
        self.trans_matrix = self.gen_transitions()
        self.init_abs_state = None
        self.state_index_set = np.arange(len(self.state_set))
        self.action_index_set = np.arange(len(self.action_set))
        self.update(initial_position, label_function)

    def update(self, position, label_function):
        abs_state = [int(position[0]//self.map_res[0]), int(position[1]//self.map_res[1])]
        initial_state = self.get_state_index(abs_state)
        self.init_abs_state = abs_state
        label_map = self.gen_labels(label_function)
        self.MDP = MDP(self.state_index_set, self.action_index_set, self.trans_matrix, label_map, initial_state)

    def gen_abs_state(self, map_range, map_res):
        xbl = 0
        xbu = int(map_range[0] / map_res[0])
        ybl = 0
        ybu = int(map_range[1] / map_res[1])
        grid_x = np.arange(xbl, xbu)
        grid_y = np.arange(ybl, ybu)
        X, Y = np.meshgrid(grid_x, grid_y)
        self.map_shape = (len(grid_x), len(grid_y))
        return np.array([X.flatten(), Y.flatten()]).T

    def gen_abs_action(self):
        # vx_set = np.array([-2, -1, 0, 1, 2])
        # vy_set = np.array([-2, -1, 0, 1, 2])
        vx_set = np.array([-1, 0, 1])
        vy_set = np.array([-1, 0, 1])
        A, B = np.meshgrid(vx_set, vy_set)
        return np.array([A.flatten(), B.flatten()]).T


    def gen_transitions(self):
        P = None
        for i in range(len(self.state_set)):
            P_s = None
            for n in range(len(self.action_set)):
                position = (i % self.map_shape[0], int(i / self.map_shape[0]))
                action = self.action_set[n]
                P_s_a = self.trans_func(position, action)
                P_s = np.vstack((P_s, P_s_a)) if P_s is not None else P_s_a
            P_s = np.expand_dims(P_s, axis=0)
            P = np.vstack((P, P_s)) if P is not None else P_s
        return P


    def gen_labels(self, label_function):
        # The setting of resolution should correspond to regions of each label
        label_map = np.array(["_"]*len(self.state_set), dtype=object)
        for n in range(len(self.state_set)):
            for region, label in label_function.items():
                xbl, xbu, ybl, ybu = region
                xbl = xbl // self.map_res[0]
                xbu = xbu // self.map_res[0]
                ybl = ybl // self.map_res[1]
                ybu = ybu // self.map_res[1]
                if xbl <= self.state_set[n, 0] < xbu and ybl <= self.state_set[n, 1] < ybu:
                    label_map[n] = label if label_map[n] == '_' else label_map[n] + label
        return label_map


    def get_state_index(self, abs_state):
        if abs_state[0] < 0 or abs_state[1] < 0 or abs_state[0] >= self.map_shape[0] or abs_state[1] >= self.map_shape[1]:
            return None
        state_index = self.state_set.tolist().index(abs_state)
        return state_index

    def get_abs_ind_state(self, position):
        abs_state = [int(position[0]//self.map_res[0]), int(position[1]//self.map_res[1])]
        state_index = self.get_state_index(abs_state)
        return state_index, abs_state

    def get_abs_state(self, position):
        if (position[0] < 0) or (position[1] < 0) or (position[0] > self.map_range[0]) or (position[1] > self.map_range[1]):
            return [-1, -1]
        return [int(position[0]//self.map_res[0]), int(position[1]//self.map_res[1])]


    def trans_func(self, position, action):
        def action_prob(action, scenario):
            # for traverse  + pedstrian scenarios
            if scenario == "traverse":
                if action == -1:
                    prob = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
                elif action == 0:
                    prob = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
                elif action == 1:
                    prob = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
                return prob

            elif scenario == "pedestrian":
                if action == -1:
                    prob = np.array([0.1, 0.7, 0.2, 0.0, 0.0])
                elif action == 0:
                    prob = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
                elif action == 1:
                    prob = np.array([0.0, 0.0, 0.2, 0.7, 0.1])
                return prob
            else:
                 # for intersection scenarios
                if action == -1:
                    prob = np.array([0.0, 0.5, 0.5, 0.0, 0.0])
                elif action == 0:
                    prob = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
                elif action == 1:
                    prob = np.array([0.0, 0.0, 0.5, 0.5, 0.0])
                return prob

        # map = self.state_set.reshape([self.map_shape[1], self.map_shape[0], 2]).transpose((1, 0, 2))
        P_sn = np.zeros(len(self.state_set)).reshape(self.map_shape)

        prob_x = action_prob(action[0], scenario=self.scenario)
        prob_y = action_prob(action[1], scenario=self.scenario)
        prob_map = np.outer(prob_x, prob_y)
        for m in range(len(prob_x)):
            for n in range(len(prob_y)):
                if (0 <= position[0] + m - 2 <= self.map_shape[0] -1) and (0 <= position[1] + n - 2 <= self.map_shape[1]-1):
                    P_sn[position[0] + m - 2, position[1] + n - 2] = prob_map[m, n]
        return P_sn.flatten(order='F')



class Abstraction_2:
    def __init__(self, map_range, map_res):
        self.map_res = map_res
        self.map_shape = None
        self.state_set = self.abs_state(map_range, map_res)
        self.action_set = self.abs_action()

    def abs_state(self, map_range, map_res):
        xbl = 0
        xbu = map_range[0]
        ybl = 0
        ybu = map_range[1]
        grid_x = np.arange(xbl, xbu, map_res[0])
        grid_y = np.arange(ybl, ybu, map_res[1])
        X, Y = np.meshgrid(grid_x, grid_y)
        self.map_shape = (len(grid_x), len(grid_y))
        return np.array([X.flatten(), Y.flatten()]).T

    def abs_action(self):
        vx_set = np.array([-1, 0, 1])
        # vy_set = np.array([-1, 0, 1])
        # vx_set = np.array([-2, -1, 0, 1, 2])
        vy_set = np.array([-2, -1, 0, 1, 2])
        A, B = np.meshgrid(vx_set, vy_set)
        return np.array([A.flatten(), B.flatten()]).T

    def get_state_index(self, abs_state):
        state_index = self.state_set.tolist().index(abs_state)
        return state_index

    def linear(self):
        # based on single integrator
        P = None
        for i in range(len(self.state_set)):
            P_s = None
            for n in range(len(self.action_set)):
                position = (i % self.map_shape[0], int(i / self.map_shape[0]))
                action = self.action_set[n]
                P_s_a = self.transition(position, action)
                P_s = np.vstack((P_s, P_s_a)) if P_s is not None else P_s_a
            P_s = np.expand_dims(P_s, axis=0)
            P = np.vstack((P, P_s)) if P is not None else P_s
        return P


    def transition(self, position, action):
        def action_prob(action, std_dev, size):
            n = int(size / 2)
            x = np.linspace(-n, n, size)
            # gaussian_array = norm.pdf(x, 0, (abs(action) + 1) * std_dev)
            gaussian_array = norm.pdf(x, action,  std_dev)
            gaussian_array /= gaussian_array.sum()
            return gaussian_array

        P_sn = np.zeros(len(self.state_set)).reshape(self.map_shape)
        prob_x = action_prob(action[0], 1.0, 5)
        prob_y = action_prob(action[1],  1.0, 5)
        prob_map = np.outer(prob_x, prob_y)
        k = int(len(prob_x) / 2)

        for m in range(len(prob_x)):
            for n in range(len(prob_y)):
                if (0 <= position[0] + m - k <= self.map_shape[0] -1) and (0 <= position[1] + n - k <= self.map_shape[1]-1):
                    P_sn[position[0] + m - k, position[1] + n - k] = prob_map[m, n]

        return P_sn.flatten(order='F')



if __name__ == '__main__':
    pcpt_range = (20, 20)
    pcpt_res = (5, 5)
    dt = 1
    initial_position = (2, 2)
    label_func = {(15, 20, 15, 20): "t",
                  (5, 15, 5, 10): "o",
                  (10, 20, 0, 20): "r"}

    abs_model = Abstraction(pcpt_range, pcpt_res, initial_position, label_func, scenario="pedestrian")
    MDP = abs_model.MDP
    #print(abs_model.MDP.transitions)
    print(abs_model.MDP.labelling)
    #print(abs_model.MDP.initial_state)