# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import opfython.math.general as g
import opfython.stream.splitter as s
from opfython.models.unsupervised import UnsupervisedOPF
from sklearn.datasets import load_digits

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

digits = load_digits()

X = digits.data
Y = digits.target

# OPF expects positive class labels rather than zero-based labels
Y += 1

X_train, X_test, Y_train, Y_test = s.split(X, Y, percentage=0.5, random_state=1)


def unsupervised_opf_clustering(opytimizer: np.ndarray) -> float:
    """Fit an unsupervised OPF and propagate labels on the shared digit split.

    Args:
        opytimizer: One-row position array containing the maximum neighbor count, truncated to an integer.

    Returns:
        One minus OPF accuracy on the held-out test split.

    """

    max_k = int(opytimizer[0][0])

    opf = UnsupervisedOPF(max_k=max_k, distance="log_squared_euclidean", pre_computed_distance=None)

    opf.fit(X_train, Y_train)

    # Labeled training data allows class predictions rather than only cluster identifiers
    opf.propagate_labels()

    preds, _ = opf.predict(X_test)

    acc = g.opf_accuracy(Y_test, preds)

    return 1 - acc


n_agents = 5
n_variables = 1

lower_bound = [1]
upper_bound = [15]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

opt = Opytimizer(space, optimizer, unsupervised_opf_clustering)

opt.start(n_iterations=3)
