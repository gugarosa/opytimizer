# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import optim
from torch.autograd import Variable

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

digits = load_digits()
X = digits.data
Y = digits.target

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.5, random_state=42)

X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
Y_train = torch.from_numpy(Y_train).long()


def fit(
    model: torch.nn.Module,
    loss: torch.nn.Module,
    opt: optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Update model parameters with one training batch.

    Args:
        model: Model whose parameters are updated in place.
        loss: Loss module comparing model outputs and target labels.
        opt: Optimizer whose gradients and state are updated.
        x: Input batch tensor.
        y: Target label tensor.

    Returns:
        Scalar loss before the parameter update.

    """

    x = Variable(x, requires_grad=False)
    y = Variable(y, requires_grad=False)

    opt.zero_grad()
    fw_x = model.forward(x)
    output = loss.forward(fw_x, y)

    output.backward()
    opt.step()

    return output.item()


def predict(model: torch.nn.Module, x_val: torch.Tensor) -> np.ndarray:
    """Predict class indices using the model's current training mode.

    Args:
        model: Trained CPU model evaluated without changing its mode.
        x_val: Validation input tensor.

    Returns:
        NumPy array of predicted class indices.

    """

    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    y_val = output.data.numpy().argmax(axis=1)

    return y_val


def logistic_regression(opytimizer: np.ndarray) -> float:
    """Train a fresh logistic classifier on the shared digit split.

    Args:
        opytimizer: Position rows containing SGD learning rate and momentum in that order.

    Returns:
        One minus validation accuracy after one hundred training epochs.

    """

    model = torch.nn.Sequential()

    n_features = 64
    n_classes = 10

    model.add_module("linear", torch.nn.Linear(n_features, n_classes, bias=False))

    batch_size = 100
    epochs = 100

    learning_rate = opytimizer[0][0]
    momentum = opytimizer[1][0]

    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for _ in range(epochs):
        cost = 0.0
        num_batches = len(X_train) // batch_size

        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += fit(model, loss, opt, X_train[start:end], Y_train[start:end])

    preds = predict(model, X_val)
    acc = np.mean(preds == Y_val)

    return 1 - acc


n_agents = 10
n_variables = 2

lower_bound = [0, 0]
upper_bound = [1, 1]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

opt = Opytimizer(space, optimizer, logistic_regression)

opt.start(n_iterations=100)
