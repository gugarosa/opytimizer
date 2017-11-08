import opytimizer.core.agent as Agent

import numpy as np

class SearchSpace(object):
    """ A SearchSpace class for running meta-heuristic optimization
        techniques.

        # Properties
            n_agents: Number of agents.
            n_variables: Number of decision variables.

        # Methods
        call(): search space's pure logic
        
        # Class Methods

        # Internal Methods
        build(input_shape)
    """

    def __init__(self, n_agents=1, 
                n_variables=1,
                **kwargs):
        # These properties will be set upon call of self.build()
        self._built = False

        allowed_kwargs = {'n_agents',
                            'n_variables'}
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
                
        if 'n_agents' in kwargs and 'n_variables' in kwargs:
            self.a = [Agent.Agent(n_variables) for _ in range(n_agents) 

        
    @property
    def built(self):
        return self._built

    @built.setter
    def built(self, value):
        self._built = value

    def call(self, inputs, **kwargs):

    """Search space's pure logic.

        # Arguments
        inputs: input tensor, list/tuple of input tensors
        **kwargs: Additional keyword arguments.

        # Returns
        A tensor of list/tuple of tensors
    """
        return inputs

    def __call__(self, inputs, **kwargs)
    """Wrapper around self.call() for handling internal logic.

        # Arguments
        inputs: input tensor, list/tuple of input tensors
        **kwargs: Additional keyword arguments.

        # Returns
        Output of the search space's `call` method.
    """

    def build(self, input_shape):
    """Creates the search space.

    # Arguments
        input_shape: opytimizer tensor or list/tuple of opytimizer
        tensors for future reference.
    """
        self.built = True
