import numpy as np
from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.pso import PSO
from opytimizer.spaces.search import SearchSpace
from sklearn.datasets import load_digits

import opfython.math.general as g
import opfython.stream.splitter as s
from opfython.models.unsupervised import UnsupervisedOPF

# Loading digits dataset
digits = load_digits()

# Gathering samples and targets
X = digits.data
Y = digits.target

# Adding 1 to labels, i.e., OPF should have labels from 1+
Y += 1

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = s.split(
    X, Y, percentage=0.5, random_state=1)


def unsupervised_opf_clustering(opytimizer):
    # Gathering hyperparams
    max_k = int(opytimizer[0][0])

    # Creates an UnsupervisedOPF instance
    opf = UnsupervisedOPF(
        max_k=max_k, distance='log_squared_euclidean', pre_computed_distance=None)

    # Fits training data into the classifier
    opf.fit(X_train, Y_train)

    # If data is labeled, one can propagate predicted labels instead of only the cluster identifiers
    opf.propagate_labels()

    # Predicts new data
    preds = opf.predict(X_test)

    # Calculating accuracy
    acc = g.opf_accuracy(Y_test, preds)

    return 1 - acc


# Creating Function's object
f = Function(pointer=unsupervised_opf_clustering)

# Number of agents
n_agents = 5

# Number of decision variables
n_variables = 1

# Number of running iterations
n_iterations = 3

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = [1]
upper_bound = [15]

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
