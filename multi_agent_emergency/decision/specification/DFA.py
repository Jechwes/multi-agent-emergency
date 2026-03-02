

class DFA:
    def __init__(self, states, alphabet, transitions, initial_state, sink_states, AP_set):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.initial_state = initial_state
        self.sink_states = sink_states
        self.AP = AP_set

    def get_alphabet(self, letter):
        # Create a tuple based on the presence of each letter in ap_set in the string 'letter'
        return tuple(ap in letter for ap in self.AP)

    def is_sink_state(self, state):
        return state in self.sink_states


