import numpy as np

import opytimizer.visualization.convergence as c
from opytimizer.utils.history import History

#
history = History()

#
history.load('pso.pkl')

#
print(history.get('best_agent', (1,)))


