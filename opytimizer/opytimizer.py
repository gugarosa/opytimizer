class Opytimizer:
    """
    """

    def __init__(self, space=None, optimizer=None, function=None):
        """
        """

        self.space = space
        self.optimizer = optimizer
        self.function = function

    def evaluate(self):
        """
        """

        for i in range(self.space.n_agents):
            print(self.function.function(self.space.agents[i].position))