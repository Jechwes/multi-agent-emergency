import networkx as nx
import itertools
import pygraphviz as pgv
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import from_agraph
from ltlf2dfa.parser.ltlf import LTLfParser
from IPython.display import Image
from specification.DFA import DFA
import graphviz
import re

class Translate:
    def __init__(self, spec, AP_set):
        self.spec = spec
        self.AP = AP_set
        self.alphabet_set = list(itertools.product([True, False], repeat=len(self.AP)))
        self.dfa = self.translate(spec)

    def translate(self, spec):
        parser = LTLfParser()
        formula = parser(spec)  # returns an LTLfFormula
        dfa_dot = formula.to_dfa()
        dfa_graph = self.to_network(dfa_dot)
        # dfa_graph = self.to_network(dfa_dot, plot_flag=True, image_flag=True)

        edges_info = {edge: dfa_graph.edges[edge] for edge in dfa_graph.edges}
        state_set = [*range(1, len(dfa_graph.nodes))]
        trans_condition = {}

        sink_states = []
        initial_state = 'init'
        for edge, _obs in edges_info.items():
            if _obs.get('label') is not None:
                obs = _obs.get('label')
                trans_condition[edge] = obs
            if (_obs.get('label') == "true") and (edge[0] == edge[1]):
                sink_states.append(edge[0])
            if (edge[0] == 'init') and (_obs.get('label') is None):
                initial_state = edge[1]

        transitions_set = self.gen_transition(trans_condition)
        dfa = DFA(states=state_set,
                  alphabet= self.alphabet_set,
                  transitions=transitions_set,
                  initial_state = initial_state,
                  sink_states= sink_states,
                  AP_set=self.AP)
        # print(dfa.transitions)
        # print(dfa.initial_state)
        # print(dfa.states)
        # print(dfa.alphabet)
        return dfa

    def get_alphabet(self, letter):
        # Create a tuple based on the presence of each letter in ap_set in the string 'letter'
        return tuple(ap in letter for ap in self.AP)

    def gen_transition(self, trans_condition):
        def parse_expression(expression):
            # Replace logical operators with Python equivalents
            expression = expression.replace("&", " and ")
            expression = expression.replace("|", " or ")
            expression = expression.replace("~", " not ")
            return expression
        def evaluate_condition(condition, AP_set, letter):
            expression = parse_expression(condition)
            # Create a dictionary to map words to their boolean values
            letter_dict = dict(zip(AP_set, letter))
            # Replace words in the expression with their boolean values
            for AP, value in letter_dict.items():
                expression = re.sub(r'\b' + AP + r'\b', str(value), expression)
            # Evaluate the final expression
            return eval(expression)

        transition_set = {}
        for edge, condition in trans_condition.items():
            for letter in self.alphabet_set:
                if condition == 'true':
                    transition_set[(edge[0], letter)] = edge[1]
                else:
                    if evaluate_condition(condition, self.AP, letter):
                            transition_set[(edge[0], letter)] = edge[1]
        return transition_set


    def to_network(self, dfa_dot, plot_flag=False, image_flag=False):
        agraph = pgv.AGraph(string=dfa_dot)
        dfa_graph = from_agraph(agraph)
        if plot_flag:
            pos = nx.spring_layout(dfa_graph)
            nx.draw(dfa_graph, pos, with_labels=True)
            plt.show()
        if image_flag:
            _graph = graphviz.Source(dfa_dot)
            output_filename = 'MONA_DFA_1'
            _graph.render(output_filename, format='png', cleanup=True)
            Image(filename=f'{output_filename}.png')
        return dfa_graph




if __name__ == "__main__":
    # The syntax of LTLf is defined in the link: http://ltlf2dfa.diag.uniroma1.it/ltlf_syntax
    # AP_set = ['n', 'v', 'i', 'g']
    # safe_frag = Translate("G(!n & !v) & G(!i U g)", AP_set)
    safe_spec = Translate("G(~i U g) & G(~v) & G(~n)", AP_set=['i', 'g', 'v', 'n'])
    # print(safe_frag.get_alphabet('r&o'))
    # scltl_frag = Translate("F(t)", ['t'])


