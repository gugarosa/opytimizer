import numpy as np

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.swarm.pso import PSO
from opytimizer.spaces.search import SearchSpace
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

# Loading digits dataset
digits = load_digits()

# Gathering samples and targets
X = digits.data
Y = digits.target

def k_means_clustering(opytimizer):
    # Gathering hyperparams
    n_clusters = int(opytimizer[0][0])

    # Instanciating an KMeans class
    kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(X)

    # Gathering predicitions
    preds = kmeans.labels_

    # Calculating adjusted rand index
    ari = metrics.adjusted_rand_score(Y, preds)

    return 1 - ari


# Creating Function's object
f = Function(pointer=k_means_clustering)

# Number of agents
n_agents = 10

# Number of decision variables
n_variables = 1

# Number of running iterations
n_iterations = 100

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = [1]
upper_bound = [100]

# Creating the SearchSpace class
s = SearchSpace(n_agents=n_agents, n_iterations=n_iterations,
                n_variables=n_variables, lower_bound=lower_bound,
                upper_bound=upper_bound)

# Hyperparameters for the optimizer
hyperparams = {
    'w': 0.7,
    'c1': 1.7,
    'c2': 1.7
}

# Creating PSO's optimizer
p = PSO(hyperparams=hyperparams)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
history = o.start()
