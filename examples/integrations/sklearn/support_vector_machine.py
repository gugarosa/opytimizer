import numpy as np
from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.pso import PSO
from opytimizer.spaces.search import SearchSpace
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, cross_val_score

# Loading digits dataset
digits = load_digits()

# Gathering samples and targets
X = digits.data
Y = digits.target

def support_vector_machine(opytimizer):
    # Gathering hyperparams
    C = opytimizer[0][0]

    # Instanciating an SVC class
    svc = svm.SVC(C=C, kernel='linear')

    # Creating a cross-validation holder
    k_fold = KFold(n_splits=5)

    # Fitting model using cross-validation
    scores = cross_val_score(svc, X, Y, cv=k_fold, n_jobs=-1)

    # Calculating scores mean
    mean_score = np.mean(scores)

    return 1 - mean_score


# Creating Function's object
f = Function(pointer=support_vector_machine)

# Number of agents
n_agents = 10

# Number of decision variables
n_variables = 1

# Number of running iterations
n_iterations = 100

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = [0.00001]
upper_bound = [10]

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
