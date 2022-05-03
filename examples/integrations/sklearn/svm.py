import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, cross_val_score

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

# Loads digits dataset
digits = load_digits()

# Gathers samples and targets
X = digits.data
Y = digits.target


def _svm(opytimizer):
    # Gathers params
    C = opytimizer[0][0]

    # Instanciating an SVC class
    svc = svm.SVC(C=C, kernel="linear")

    # Creates a cross-validation holder
    k_fold = KFold(n_splits=5)

    # Fitting model using cross-validation
    scores = cross_val_score(svc, X, Y, cv=k_fold, n_jobs=-1)

    # Calculates scores mean
    mean_score = np.mean(scores)

    return 1 - mean_score


# Number of agents and decision variables
n_agents = 10
n_variables = 1

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [0.000001]
upper_bound = [10]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = Function(_svm)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=100)
