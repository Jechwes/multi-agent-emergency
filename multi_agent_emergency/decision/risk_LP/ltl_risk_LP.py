#!/usr/bin/env python
import gurobipy as grb
import numpy as np
import time

class Risk_LTL_LP:

    def __init__(self):
        self.state_num = None
        self.action_num = None

    def solve(self, P, c_map, initial_state, accept_states, risk_th, initial_guess=None):
        self.action_num = P.shape[1]  # number of actions
        self.state_num = P.shape[0]  # number of states not in T
        S0 = initial_state  # The initial state
        gamma = 0.5 # discount factor
        model = grb.Model("risk_lp")
        y = model.addVars(self.state_num, self.action_num, vtype=grb.GRB.CONTINUOUS, name='x') # occupation measure
        z = model.addVars(1, vtype=grb.GRB.CONTINUOUS, name='z')
        pi = model.addVars(self.state_num, vtype=grb.GRB.CONTINUOUS, name='strategy')
        # Set initial values for warm start if provided
        if initial_guess is not None:
            for s in range(self.state_num):
                for a in range(self.action_num):
                    y[s, a].start = initial_guess[s, a]

        x = []
        for s in range(self.state_num):  # compute occupation
            x += [grb.quicksum(y[s, a] for a in range(self.action_num))]

        xi = []
        for sn in range(self.state_num):  # compute incoming occupation
            # from s to s' sum_a x(s, a) P(s,a,s)'
            xi += [gamma * grb.quicksum(
                   y[s, a] * P[s][a][sn] for a in range(self.action_num) for s in range(self.state_num))]

        lhs = [x[i] - xi[i] for i in range(len(xi))]
        rhs = [0] * self.state_num
        rhs[S0] = 1
        for i in range(self.state_num):
            model.addConstr(lhs[i] == rhs[i])

        obj = 10 * grb.quicksum(y[s, a] * P[s][a][sn]
                           for a in range(self.action_num)
                           for s in range(self.state_num)
                           for sn in accept_states) - z[0]**2

        model.addConstr(grb.quicksum(pi[s] * c_map[s] for s in range(self.state_num)) <= risk_th + z[0])
        for s in range(self.state_num):
            max_expr = grb.max_([y[s, a] for a in range(self.action_num)])
            model.addConstr(pi[s] == max_expr)
        model.setObjective(obj, grb.GRB.MAXIMIZE)

        time_start = time.time()
        model.optimize()
        time_end = time.time()
        print("calculate time: ", time_end - time_start)
        sol = model.getAttr('x', y)
        policy_map = model.getAttr('x', pi)
        relax = model.getAttr('x', z)
        print("relaxation: ", relax)
        return sol, policy_map


    def extract(self, occup_dict):
        strategy = np.zeros(self.state_num)
        risk_field = np.zeros(self.state_num)
        for occ in occup_dict.items():
            state = occ[0][0]
            action = occ[0][1]
            prob = occ[1]
            if prob > risk_field[state]:
                risk_field[state] = np.log10(10 * prob + 1)
                strategy[state] = int(action)
        return strategy, risk_field/max(risk_field)


if __name__ == '__main__':
    P = np.array([[[0.2, 0, 0, 0.8, 0, 0], [0, 0.2, 0, 0, 0.8, 0], [0, 0, 0.2, 0, 0, 0.8],
          [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], # action: up
         [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
          [0.8, 0, 0, 0.2, 0, 0], [0, 0.8, 0, 0, 0, 0.2], [0, 0, 0.8, 0, 0, 0.2]],  # action: down
         [[0.2, 0.8, 0, 0, 0, 0], [0, 0.2, 0.8, 0, 0, 0], [0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0.2, 0.8, 0], [0, 0, 0, 0, 0.2, 0.8], [0, 0, 0, 0, 0, 1]],  # action: right
         [[1, 0, 0, 0, 0, 0], [0.8, 0.2, 0, 0, 0, 0], [0, 0.8, 0.2, 0, 0, 0],
          [0, 0, 0, 1, 0, 0], [0, 0, 0, 0.8, 0.2, 0], [0, 0, 0, 0, 0.8, 0.2]],  # action: left
          ]) # stochastic transition
    c_map = [1, 1, 1, 1, -10, 10]  # cost map
    LP_prob = Risk_LTL_LP()
    occ_dict = LP_prob.solve(P, c_map, 0)
    strategy, risk_field = LP_prob.extract(occ_dict)
    print(strategy)
    print(risk_field)