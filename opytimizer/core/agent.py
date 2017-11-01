import numpy as np

class Agent(object):
    """
    A agent class for all meta-heuristic optimization techniques. Agents have the following properties:
        n: number of decision variables
        x: n-dimensional array of position values
        fit: agent's fitness value
    """

    def __init__(self, n):
        # Return an Agent object with n-dimensions
        self.n = n
        self.x = np.zeros(n)
        self.fit = 0

    def CheckLimits(self, LB, UB):
        # Check an Agent limits based on lower and upper bounds
        for i in range(self.n):
            if self.x[i] < LB[i]:
                self.x[i] = LB[i]
            elif self.x[i] > UB[i]:
                self.x[i] = UB[i]