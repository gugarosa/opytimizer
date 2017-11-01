import opytimizer.core.agent as Agent

import numpy as np

class SearchSpace(object):
    '''
    A SearchSpace class for running meta-heuristic optimization techniques. SearchSpace have the following properties:
        m : number of agents
        n : number of decision variables
        custom variables depending on selected algorithm
    '''

    def __init__(self, m, n):
        # Return a SearchSpace object with 'm' n-dimensional agents
        self.a = [Agent.Agent(n) for _ in range(m)]

    def initialize(self, id, model_file):
        # Initializes a SearchSpace object and its variables depending on the chosen algorithm.
        # Note that it uses a model file in order to parse its configurations.
        if id == 'PSO':
            self.w = 1