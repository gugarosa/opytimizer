# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import opfython.math.general as g
import opfython.stream.splitter as s
from opfython.models.supervised import SupervisedOPF
from sklearn.datasets import load_digits

from opytimizer import Opytimizer
from opytimizer.optimizers.boolean import BPSO
from opytimizer.spaces import BooleanSpace

digits = load_digits()

X = digits.data
Y = digits.target

# OPF expects positive class labels rather than zero-based labels
Y += 1

X_train, X_val, Y_train, Y_val = s.split(X, Y, percentage=0.5, random_state=1)


def supervised_opf_feature_selection(opytimizer: np.ndarray) -> float:
    """Fit a supervised OPF on the selected digit features.

    Args:
        opytimizer: Binary position array selecting columns from the shared digit dataset.

    Returns:
        One minus OPF accuracy on the held-out validation split.

    """

    features = opytimizer[:, 0].astype(bool)

    X_train_selected = X_train[:, features]
    X_val_selected = X_val[:, features]

    opf = SupervisedOPF(distance="log_squared_euclidean", pre_computed_distance=None)

    opf.fit(X_train_selected, Y_train)

    preds = opf.predict(X_val_selected)

    acc = g.opf_accuracy(Y_val, preds)

    return 1 - acc


n_agents = 5
n_variables = 64

space = BooleanSpace(n_agents, n_variables)
optimizer = BPSO()

opt = Opytimizer(space, optimizer, supervised_opf_feature_selection)

opt.start(n_iterations=3)
