import numpy as np

class Risk_LTL_DP:
    def __init__(self):
        self.state_num = None
        self.action_num = None
        self.T = 20 # finite horizon
        self.gamma = 0.95 # discounting factor
        self.K = 0.01 # risk tuning parameter
        self.computed_policy = None

    def solve(self, P, cost_map, ego_prod_state_index, accepting_states, risk_th, soft_risk_th=None):
           
        self.state_num = P.shape[0]  # number of states not in T
        self.action_num = P.shape[1]  # number of actions
        V = np.zeros(self.state_num)
        valid_accepting = [s for s in accepting_states if s < self.state_num]
        V[valid_accepting] = 1.0 # accepting states set to 1
        
        risk_costs= self.K * cost_map

        for t in range(self.T):
            V_adjusted = V - risk_costs
            for acc in valid_accepting:
                V_adjusted[acc] = 1.0

            expected_values = np.dot(P, V_adjusted) 
            Q_values = self.gamma * expected_values
            V = np.max(Q_values, axis=1) #V (t+1)
            V[valid_accepting] = 1.0

        V_adjusted = V - risk_costs
        for acc in valid_accepting:
            V_adjusted[acc] = 1.0
        expected_values = np.dot(P, V_adjusted)
        Q_values = self.gamma * expected_values

        # Make sure that the cars do not drift from accepting state
        for acc in valid_accepting:
            self_trans_probs = P[acc, :, acc]
            stay_actions = self_trans_probs > 0.99
            if np.any(stay_actions):
                Q_values[acc, stay_actions] += 1e-6

        best_actions = np.argmax(Q_values, axis=1)
            
        # Store as dense array for compatibility
        self.computed_policy = best_actions
            
        # Return V (as dummy occupation measure) and Policy Map
        return V, best_actions

    def extract(self, occ_measure=None):
        if self.computed_policy is None:
            return np.array([]), None
            
        return self.computed_policy, None

if __name__ == "__main__":
    if __name__ == "__main__":
        print("--- Running Risk_LTL_DP Test ---")
        
        # 1. Setup simple grid world
        # 3 States: 0(Start), 1(Risk), 2(Goal)
        # 2 Actions: 0(To Risk), 1(To Goal)
        
        num_states = 3
        num_actions = 2
        
        # P matrix shape: (States, Actions, Next_States)
        P = np.zeros((num_states, num_actions, num_states))
        
        # State 0 Transitions
        # Action 0 -> 100% to State 1 (Risky)
        P[0, 0, 1] = 1.0 
        # Action 1 -> 100% to State 2 (Safe Goal)
        P[0, 1, 2] = 1.0 
        
        # State 1 (Risky) - eventually goes to goal
        P[1, :, 2] = 1.0 
        
        # State 2 (Goal) - Absorbing
        P[2, :, 2] = 1.0
        
        # 2. Costs
        # State 1 has cost 50 (High Risk), others 0
        cost_map = np.array([0.0, 50.0, 0.0])
        
        # 3. Solver
        solver = Risk_LTL_DP()
        solver.K = 0.05
        solver.gamma = 0.8
        
        print(f"Cost Map: {cost_map}")
        print(f"K (Penalty Factor): {solver.K}")
        
        V, policy = solver.solve(P, cost_map, 0, [2], 0.1)
        
        print("\n--- Results ---")
        print(f"Value Function V: {V}")
        print(f"Policy: {policy}")
        
        # Explanation
        print("\nExplanation:")
        print(f"V[0] = {V[0]:.4f}")
        if policy[0] == 1:
            print("Robot chose Action 1 (Direct to Safe Goal) -> CORRECT")
        else:
            print("Robot chose Action 0 (Via Risky State) -> INCORRECT (Too risky)")
