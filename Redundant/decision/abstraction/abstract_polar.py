#!/usr/bin/env python
import gurobipy as grb
import numpy as np
import math
from abstraction.MDP import MDP
from scipy.stats import norm

class Abstraction:

    def __init__(self, origin_point, lane_start, lane_amount, cell_width, angle_step, initial_position, label_function):
        self.origin_point = origin_point
        self.lane_start = lane_start
        self.lane_amount = lane_amount
        self.cell_width = cell_width
        self.angle_step = angle_step
        
        # Grid dimensions: (Rows=Radii, Cols=Angles)
        num_sectors = int(360 / angle_step)
        self.map_shape = (lane_amount, num_sectors)

        self.state_set = self.gen_abs_state()
        self.action_set = self.gen_abs_action()
        self.trans_matrix = self.gen_transitions()
        
        self.init_abs_state = None
        self.state_index_set = np.arange(len(self.state_set))
        self.action_index_set = np.arange(len(self.action_set))
        
        self.update(initial_position, label_function)

    def update(self, position, label_function):
        # Position (x, y) to abstract state (r, theta)
        abs_state = self.get_abs_state(position)
        initial_state = self.get_state_index(abs_state)
        self.init_abs_state = abs_state
        
        label_map = self.gen_labels(label_function)
        self.MDP = MDP(self.state_index_set, self.action_index_set, self.trans_matrix, label_map, initial_state)

    def gen_abs_state(self):
        # generate all (r, theta) pairs
        grid_r = np.arange(self.map_shape[0])
        grid_theta = np.arange(self.map_shape[1])
        R, T = np.meshgrid(grid_r, grid_theta, indexing='ij') 
        return np.array([R.flatten(), T.flatten()]).T

    def gen_abs_action(self):
        vr_set = np.array([-1, 0, 1]) # -1: In, 0: Stay, 1: Out
        vt_set = np.array([-1, 0]) # -1 (CCW), 0 (Stay). 1 (CW) (1 not used as no CW on roundabout).
        A, B = np.meshgrid(vr_set, vt_set)
        return np.array([A.flatten(), B.flatten()]).T

    def gen_transitions(self):
        P = None
        for i in range(len(self.state_set)):
            P_s = None
            current_state = self.state_set[i] # [r_idx, theta_idx]
            
            for n in range(len(self.action_set)):
                action = self.action_set[n]
                P_s_a = self.trans_func(current_state, action)
                P_s = np.vstack((P_s, P_s_a)) if P_s is not None else P_s_a
            
            P_s = np.expand_dims(P_s, axis=0)
            P = np.vstack((P, P_s)) if P is not None else P_s
        return P

    def gen_labels(self, label_function):
        # The setting of resolution should correspond to regions of each label
        label_map = np.array(["_"]*len(self.state_set), dtype=object)
        for n in range(len(self.state_set)):
            state_key = (int(self.state_set[n][0]), int(self.state_set[n][1]))
            if state_key in label_function:
                 label = label_function[state_key]
                 if label_map[n] == '_':
                     label_map[n] = label
                 else:
                     label_map[n] += label       
        return label_map

    def get_state_index(self, abs_state):
        if abs_state[0] < 0 or abs_state[0] >= self.map_shape[0]: # check Radius bounds
            return None
        
        r_idx = int(abs_state[0])
        t_idx = int(abs_state[1]) % self.map_shape[1] # wrap angle
        
        return r_idx * self.map_shape[1] + t_idx

    def get_abs_state(self, position):
        # position: (x, y) or object with x,y
        if hasattr(position, 'x'):
            pos_x = position.x
            pos_y = position.y
        else:
            pos_x = position[0]
            pos_y = position[1]

        if hasattr(self.origin_point, 'x'):
            org_x = self.origin_point.x
            org_y = self.origin_point.y
        else:
            org_x = self.origin_point[0]
            org_y = self.origin_point[1]

        dx = pos_x - org_x
        dy = pos_y - org_y

        r = math.sqrt(dx**2 + dy**2)
        theta_rad = math.atan2(dy, dx)
        theta_deg = math.degrees(theta_rad)
        if theta_deg < 0:
            theta_deg += 360

        # Calculate indices
        if r < self.lane_start:
            return [-1, -1] # Inner dead zone
        
        r_idx = int((r - self.lane_start) / self.cell_width)
        theta_idx = int(theta_deg / self.angle_step)
        
        # Check bounds
        if r_idx >= self.lane_amount:
            return [-1, -1] # Outside grid
        
        # Handle 360 case (rare edge case where theta_deg == 360)
        theta_idx = theta_idx % self.map_shape[1]
            
        return [r_idx, theta_idx]
    
    def get_abs_ind_state(self, position):
        abs_state = self.get_abs_state(position)
        state_index = self.get_state_index(abs_state)
        return state_index, abs_state

    def trans_func(self, position, action):
        # position: [r, theta] (indices)
        # action: [dr, dt]
     
        def action_prob(act):            
            if act == -1:
                prob = np.array([0.0, 0.5, 0.5, 0.0, 0.0])
            elif act == 0:
                prob = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
            elif act == 1:
                prob = np.array([0.0, 0.0, 0.5, 0.5, 0.0])
            return prob

        kernel_range = np.arange(-2, 3) 
        prob_r = action_prob(action[0])
        prob_t = action_prob(action[1])
        prob_map = np.outer(prob_r, prob_t) # 5x5 kernel
        
        P_sn = np.zeros(self.map_shape) # (n_lanes, n_sectors)
        
        # Iterate over the kernel
        for mr in range(len(prob_r)):     # relative row (-2..2 mapped to 0..4)
            for mt in range(len(prob_t)): # relative col (-2..2 mapped to 0..4)
                
                dr = kernel_range[mr]
                dt = kernel_range[mt]
                
                next_r = int(position[0] + dr)
                next_t = int(position[1] + dt)
                
                if 0 <= next_r < self.map_shape[0]:
                    # Wrap Angle
                    next_t = next_t % self.map_shape[1]
                    P_sn[next_r, next_t] += prob_map[mr, mt]

        return P_sn.flatten(order='C')


if __name__ == '__main__':
    # Simple test for the new class
    origin = (0, 0)
    lane_start = 10
    lane_amount = 3
    cell_width = 4
    angle_step = 90
    initial_pos = (14, 0) # Should be r=1 (14-10=4, 4/4=1), theta=0
    
    # Label: (1, 0) is target
    labels = {(1, 0): "t"}
    
    abs_model = Abstraction(origin, lane_start, lane_amount, cell_width, angle_step, initial_pos, labels, scenario="test")
    MDP = abs_model.MDP
    print(f"Initial State Index: {MDP.initial_state}")
    print(f"Labels: {MDP.labelling}")