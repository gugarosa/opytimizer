# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

# Keep the synthetic regression dataset reproducible across runs
torch.manual_seed(42)

# Reuse the same training samples for every objective evaluation
X = torch.linspace(-1, 1, 101)
Y = 2 * X + torch.randn(X.size()) * 0.33


def fit(
    model: torch.nn.Module,
    loss: torch.nn.Module,
    opt: optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Update model parameters with one regression batch.

    Args:
        model: Model whose parameters are updated in place.
        loss: Loss module comparing model outputs and regression targets.
        opt: Optimizer whose gradients and state are updated.
        x: One-dimensional input batch tensor.
        y: Regression target tensor.

    Returns:
        Scalar loss before the parameter update.

    """

    x = Variable(x, requires_grad=False)
    y = Variable(y, requires_grad=False)

    opt.zero_grad()
    fw_x = model.forward(x.view(len(x), 1)).squeeze()
    output = loss.forward(fw_x, y)

    output.backward()
    opt.step()

    return output.item()


def linear_regression(opytimizer: np.ndarray) -> float:
    """Train a fresh linear regressor on the shared synthetic dataset.

    Args:
        opytimizer: Position rows containing SGD learning rate and momentum in that order.

    Returns:
        Mean batch loss from the final training epoch.

    """

    model = torch.nn.Sequential()
    model.add_module("linear", torch.nn.Linear(1, 1, bias=False))

    batch_size = 10
    epochs = 100

    learning_rate = opytimizer[0][0]
    momentum = opytimizer[1][0]

    loss = torch.nn.MSELoss(reduction="mean")
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for _ in range(epochs):
        cost = 0.0
        num_batches = len(X) // batch_size

        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += fit(model, loss, opt, X[start:end], Y[start:end])

    final_cost = cost / num_batches

    return final_cost


n_agents = 10
n_variables = 2

lower_bound = [0, 0]
upper_bound = [1, 1]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

opt = Opytimizer(space, optimizer, linear_regression)

opt.start(n_iterations=100)
