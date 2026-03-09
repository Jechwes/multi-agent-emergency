import networkx as nx


class MDP:
    def __init__(self, states, actions, transitions, labels_map, initial_state):
        """
        states: Set of all states
        actions: Set of all actions
        transitions: Dictionary mapping (state, action) to a array of probability with sum 1
        labels: Dictionary mapping each state to a label
        initial_state: The initial state of the MDP
        """
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.labelling = labels_map
        self.initial_state = initial_state

    def get_trans_prob(self, state, action):
        return self.transitions.get((state, action))

    # def structure_plot(self):
    #     G = nx.DiGraph() #Todo



if __name__ == "__main__":
    mdp = MDP(states = ['s1', 's2'],
        actions= ['u1', 'u2'],
        labels = [' ', 'obstacle', 'target'],
        transitions={('s1', 'u1'): [0, 1],
                     ('s1', 'u2'): [0.1, 0.9],
                     ('s2', 'u1'): [0.5, 0.5],
                     ('s2', 'u2'): [0.4, 0.6],},
        initial_state = 's1')
    prob_dist = mdp.get_trans_prob('s1', 'u2')
    print(prob_dist)