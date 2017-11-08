import numpy as np

class Agent(object):
    """A agent class for all meta-heuristic optimization techniques.
	
		# Arguments
        	n: number of decision variables
        	x: n-dimensional array of position values
        	fit: agent's fitness value
    """

    def __init__(self, n=1,
				 **kwargs):
		super(Agent, self).__init__(**kwargs)
        self.n = n
        self.x = np.zeros(n)
        self.fit = 0

    def check_limits(self, LB, UB):
		x = self.x
        for i in range(self.n):
            if x[i] < LB[i]:
                x[i] = LB[i]
            elif x[i] > UB[i]:
                x[i] = UB[i]
	return x
