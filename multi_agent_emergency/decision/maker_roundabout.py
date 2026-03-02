from decision.risk_LP.ltl_risk_DP import Risk_LTL_DP
from decision.risk_LP.prod_auto import Product
from abstraction.prod_MDP import Prod_MDP
from decision.abstraction.MDP import MDP
import numpy as np
import copy

class Risk_LTL:

    def __init__(self, abs_model, MDP, DFA_safe, DFA_sc, cost_map):
        self.abs_model = abs_model
        self.mdp = MDP
        self.DFA_safe = DFA_safe
        self.DFA_sc = DFA_sc
        self.cost_func = cost_map
        self.LP_prob = Risk_LTL_DP()
        self.prod_auto = Product(self.mdp, self.DFA_sc, self.DFA_safe) # Directly use MDP, no Prod_MDP wrapper needed for single agent

    def update(self, ego_prod_state, abs_state_sys_index, risk_th):
        P_matrix = self.prod_auto.prod_transitions
        cost_map = self.prod_auto.gen_cost_map(self.cost_func)
        ego_prod_state_index, ego_prod_state = self.prod_auto.update_prod_state(abs_state_sys_index, ego_prod_state)     
        occ_measure, policy_map = self.LP_prob.solve(P_matrix, cost_map, ego_prod_state_index,
                                    self.prod_auto.accepting_states, risk_th, None)
        risk = self.cal_risk(policy_map, cost_map)
        optimal_policy, Z = self.LP_prob.extract(occ_measure)
        decision_index = optimal_policy[ego_prod_state_index]
        return ego_prod_state, int(decision_index), optimal_policy, risk

    def get_opt_path(self, ego_prod_state, optimal_policy, ego_abs_state):
        _ego_prod_state = copy.deepcopy(ego_prod_state)
        _ego_abs_state = copy.deepcopy(ego_abs_state)
        opt_path = [_ego_abs_state]
        for n in range(6):
            _ego_abs_state_index = self.abs_model.get_state_index(_ego_abs_state)
            if _ego_abs_state_index is None:
                return opt_path
            
            _ego_prod_state_index, _ego_prod_state = self.prod_auto.update_prod_state(_ego_abs_state_index, _ego_prod_state)
                                                                                         
            _decision_index = optimal_policy[_ego_prod_state_index]
            _opt_action = self.abs_model.action_set[int(_decision_index)]
            _ego_abs_state = (_ego_abs_state + _opt_action).tolist()
            opt_path.append(_ego_abs_state)
        return opt_path

    def cal_risk(self, policy_map, cost_map):

        risk = 0
        for n in range(len(policy_map)):
            risk += policy_map[n] * cost_map[n]/1.5
        return risk