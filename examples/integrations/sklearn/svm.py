# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, cross_val_score

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

digits = load_digits()

X = digits.data
Y = digits.target


def _svm(opytimizer: np.ndarray) -> float:
    C = opytimizer[0][0]

    svc = svm.SVC(C=C, kernel="linear")

    k_fold = KFold(n_splits=5)

    scores = cross_val_score(svc, X, Y, cv=k_fold, n_jobs=-1)

    mean_score = np.mean(scores)

    return 1 - mean_score


n_agents = 10
n_variables = 1

lower_bound = [0.000001]
upper_bound = [10]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

opt = Opytimizer(space, optimizer, _svm)

opt.start(n_iterations=100)
