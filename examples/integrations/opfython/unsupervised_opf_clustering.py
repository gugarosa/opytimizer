import opfython.math.general as g
import opfython.stream.splitter as s
from opfython.models.unsupervised import UnsupervisedOPF
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

# Adding 1 to labels, i.e., OPF should have labels from 1+
Y += 1

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = s.split(
    X, Y, percentage=0.5, random_state=1)


def unsupervised_opf_clustering(opytimizer):
    # Gathers parameters from Opytimizer
    # Pay extremely attention to their order when declaring due to their bounds
    max_k = int(opytimizer[0][0])

    # Creates an UnsupervisedOPF instance
    opf = UnsupervisedOPF(
        max_k=max_k, distance='log_squared_euclidean', pre_computed_distance=None)

    # Fits training data into the classifier
    opf.fit(X_train, Y_train)

    # If data is labeled, one can propagate predicted labels instead of only the cluster identifiers
    opf.propagate_labels()

    # Predicts new data
    preds, _ = opf.predict(X_test)

    # Calculates accuracy
    acc = g.opf_accuracy(Y_test, preds)

    return 1 - acc


# Number of agents and decision variables
n_agents = 5
n_variables = 1

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [1]
upper_bound = [15]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = Function(unsupervised_opf_clustering)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=3)
