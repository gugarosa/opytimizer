# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import torch
import torchvision
from learnergy.models.bernoulli import RBM

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

train = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)


def rbm(opytimizer: np.ndarray) -> torch.Tensor:
    """Train a fresh RBM on MNIST and return its reconstruction error.

    Args:
        opytimizer: Position rows containing learning rate, momentum, and weight decay in that order.

    Returns:
        Reconstruction error after five epochs on the shared training dataset.

    """

    lr = opytimizer[0][0]
    momentum = opytimizer[1][0]
    decay = opytimizer[2][0]

    model = RBM(
        n_visible=784,
        n_hidden=128,
        steps=1,
        learning_rate=lr,
        momentum=momentum,
        decay=decay,
        temperature=1,
        use_gpu=False,
    )

    error, _ = model.fit(train, batch_size=128, epochs=5)

    return error


n_agents = 10
n_variables = 3

lower_bound = [0, 0, 0]
upper_bound = [1, 1, 1]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

opt = Opytimizer(space, optimizer, rbm)

opt.start(n_iterations=10)
