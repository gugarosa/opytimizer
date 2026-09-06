# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

digits = load_digits()

X = digits.data
Y = digits.target


def k_means_clustering(opytimizer: np.ndarray) -> float:
    """Cluster the shared digit dataset using a candidate number of clusters.

    Args:
        opytimizer: One-row position array containing the cluster count, truncated to an integer.

    Returns:
        One minus the adjusted Rand index against the known digit labels.

    """

    n_clusters = int(opytimizer[0][0])

    kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(X)

    preds = kmeans.labels_

    ari = metrics.adjusted_rand_score(Y, preds)

    return 1 - ari


n_agents = 10
n_variables = 1

lower_bound = [1]
upper_bound = [100]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

opt = Opytimizer(space, optimizer, k_means_clustering)

opt.start(n_iterations=100)
