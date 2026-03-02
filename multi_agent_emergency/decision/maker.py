from decision.risk_LP.ltl_risk_DP import Risk_LTL_DP
from decision.risk_LP.prod_auto import Product
from abstraction.prod_MDP import Prod_MDP
from decision.abstraction.MDP import MDP
import numpy as np
import copy

def None_MDP():
    state_set = range(1)
    action_set = [0]
    transitions = np.array([[[1]]])
    initial_state = 0
    mdp_env = MDP(state_set, action_set, transitions, ['_'], initial_state)
    return mdp_env

class Risk_LTL:

    def __init__(self, abs_model, MDP_sys, MDP_env, DFA_safe, DFA_sc, cost_map):
        self.abs_model = abs_model
        self.mdp_sys = MDP_sys
        self.mdp_env = MDP_env
        self.DFA_safe = DFA_safe
        self.DFA_sc = DFA_sc
        self.cost_func = cost_map
        self.LP_prob = Risk_LTL_DP()
        self.mdp_prod = Prod_MDP(self.mdp_sys, self.mdp_env) if self.mdp_env is not None \
            else Prod_MDP(self.mdp_sys, None_MDP())
        self.prod_auto = Product(self.mdp_prod.MDP, self.DFA_sc, self.DFA_safe)

    def update(self, ego_prod_state, abs_state_sys_index, env_abs_state, risk_th):
        P_matrix = self.prod_auto.prod_transitions
        cost_map = self.prod_auto.gen_cost_map(self.cost_func)

        state_sys_env_index = self.mdp_prod.get_prod_state_index((abs_state_sys_index, env_abs_state))
        ego_prod_state_index, ego_prod_state = self.prod_auto.update_prod_state(state_sys_env_index, ego_prod_state)
        occ_measure, policy_map = self.LP_prob.solve(P_matrix, cost_map, ego_prod_state_index,
                                    self.prod_auto.accepting_states, risk_th, None)
        risk = self.cal_risk(policy_map, cost_map)
        optimal_policy, Z = self.LP_prob.extract(occ_measure)
        decision_index = optimal_policy[ego_prod_state_index]
        return ego_prod_state, int(decision_index), optimal_policy, risk

    def offline_update(self, ego_prod_state, optimal_policy, abs_state_sys_index, env_abs_state):
        state_sys_env_index = self.mdp_prod.get_prod_state_index((abs_state_sys_index, env_abs_state))
        ego_prod_state_index, ego_prod_state = self.prod_auto.update_prod_state(state_sys_env_index, ego_prod_state)
        decision_index = optimal_policy[ego_prod_state_index]
        return int(decision_index), ego_prod_state

    def offline_update_2(self, optimal_policy, ego_abs_state, abs_state_sys_index, env_abs_state):
        state_sys_env_index = self.mdp_prod.get_prod_state_index((abs_state_sys_index, env_abs_state))
        ego_prod_state_index, self.ego_prod_state = self.prod_auto.update_prod_state(state_sys_env_index, self.ego_prod_state)
        decision_index = optimal_policy[ego_prod_state_index]
        opt_action = self.abs_model.action_set[decision_index]
        return int(decision_index)

    def get_opt_path(self, ego_prod_state, optimal_policy, ego_abs_state, env_abs_state):
        _ego_prod_state = copy.deepcopy(ego_prod_state)
        _ego_abs_state = copy.deepcopy(ego_abs_state)
        opt_path = [_ego_abs_state]
        for n in range(6):
            _ego_abs_state_index = self.abs_model.get_state_index(_ego_abs_state)
            if _ego_abs_state_index is None:
                return opt_path
            _prod_sys_env_index = self.mdp_prod.get_prod_state_index((_ego_abs_state_index, env_abs_state))
            _ego_prod_state_index, _ego_prod_state = self.prod_auto.update_prod_state(_prod_sys_env_index,
                                                                                         _ego_prod_state)
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