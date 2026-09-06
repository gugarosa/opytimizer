# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import torch
import torchvision
from learnergy.models.bernoulli import DropoutRBM

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

train = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)


def dropout_rbm(opytimizer: np.ndarray) -> torch.Tensor:
    """Train a fresh dropout RBM on MNIST and return its reconstruction error.

    Args:
        opytimizer: One-row position array containing the dropout probability.

    Returns:
        Reconstruction error after five epochs on the shared training dataset.

    """

    dropout = opytimizer[0][0]

    model = DropoutRBM(
        n_visible=784,
        n_hidden=128,
        steps=1,
        learning_rate=0.1,
        momentum=0,
        decay=0,
        temperature=1,
        dropout=dropout,
        use_gpu=False,
    )

    error, _ = model.fit(train, batch_size=128, epochs=5)

    return error


n_agents = 5
n_variables = 1

lower_bound = [0]
upper_bound = [1]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

opt = Opytimizer(space, optimizer, dropout_rbm)

opt.start(n_iterations=5)
