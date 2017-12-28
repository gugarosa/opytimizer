""" This is the search space's structure and its basic functions module.
"""

import json

import opytimizer.core.agent as Agent
import opytimizer.core.function as Function
import opytimizer.optimizers.pso as PSO


class SearchSpace(object):
    """ A SearchSpace class for running meta-heuristic optimization
        techniques.

        # Argument
            model_path: JSON model file containing all the needed information
            to create a Search Space.

        # Properties
            n_agents: Number of agents in the search space.
            agent: List of agents.
            optimizer: Choosen optimizer algorithm.
            function: Function object to be evaluated.
            hyperparams: Search space-related hyperparams.

        # Methods

        # Class Methods

        # Internal Methods
    """

    def __init__(self, **kwargs):
        # These properties will be set upon call of self.build()
        self._built = False

        allowed_kwargs = {'model_path'
                          }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        if 'model_path' in kwargs:
            model_path = kwargs['model_path']
            with open(model_path) as json_file:
                model = json.load(json_file)
        else:
            raise Exception(
                'Json file is missing. Please include the argument model_path.')

        # Gathering JSON keywords
        n_agents = model['n_agents']
        n_variables = model['agent']['n_variables']
        n_dimensions = model['agent']['n_dimensions']
        optimizer = model['optimizer']['algorithm']
        optimizer_hyperparams = model['optimizer']['hyperparams']
        function = model['function']['expression']
        lower_bound = model['function']['lower_bound']
        upper_bound = model['function']['upper_bound']
        hyperparams = model['hyperparams']

        # Applying variables to their corresponding creations
        self.n_agents = n_agents
        self.agent = [Agent.Agent(n_variables=n_variables,
                                  n_dimensions=n_dimensions) for _ in range(n_agents)]
        if optimizer == 'PSO':
            self.optimizer = PSO.PSO(hyperparams=optimizer_hyperparams)
        self.function = Function.Function(
            expression=function, lower_bound=lower_bound, upper_bound=upper_bound)
        self.hyperparams = hyperparams

    def evaluate(self):
        for i in range(self.n_agents):
            fitness = self.function.evaluate(self.agent[i])
        