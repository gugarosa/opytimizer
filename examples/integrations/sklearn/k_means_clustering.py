import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

# Loading digits dataset
digits = load_digits()

# Gathers samples and targets
X = digits.data
Y = digits.target

def k_means_clustering(opytimizer):
    # Gathers params
    n_clusters = int(opytimizer[0][0])

    # Instanciating an KMeans class
    kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(X)

    # Gathers predicitions
    preds = kmeans.labels_

    # Calculates adjusted rand index
    ari = metrics.adjusted_rand_score(Y, preds)

    return 1 - ari


# Number of agents and decision variables
n_agents = 10
n_variables = 1

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [1]
upper_bound = [100]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = Function(k_means_clustering)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=100)
