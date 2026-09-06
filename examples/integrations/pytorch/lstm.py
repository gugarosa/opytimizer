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

X_train = X_train.reshape(-1, 8, 8)
X_val = X_val.reshape(-1, 8, 8)

# LSTM expects sequence length before the batch and feature axes
X_train = np.swapaxes(X_train, 0, 1)
X_val = np.swapaxes(X_val, 0, 1)

X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
Y_train = torch.from_numpy(Y_train).long()


class LSTM(torch.nn.Module):
    """Classify image-row sequences with an LSTM and a linear output layer.

    """

    def __init__(self, n_features: int, n_hidden: int, n_classes: int) -> None:
        """Allocate a recurrent layer and its classification output.

        Args:
            n_features: Number of input features per sequence step.
            n_hidden: Number of recurrent hidden units.
            n_classes: Number of output classes.

        """

        super(LSTM, self).__init__()

        self.n_hidden = n_hidden
        self.lstm = torch.nn.LSTM(n_features, n_hidden)
        self.linear = torch.nn.Linear(n_hidden, n_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[1]

        # Independent batches start from zero rather than sharing recurrent state
        h0 = Variable(torch.zeros([1, batch_size, self.n_hidden]), requires_grad=False)
        c0 = Variable(torch.zeros([1, batch_size, self.n_hidden]), requires_grad=False)

        fx, _ = self.lstm.forward(x, (h0, c0))

        return self.linear.forward(fx[-1])


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
        x: Sequence-first input batch tensor.
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
        x_val: Sequence-first validation input tensor.

    Returns:
        NumPy array of predicted class indices.

    """

    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    y_val = output.data.numpy().argmax(axis=1)

    return y_val


def lstm(opytimizer: np.ndarray) -> float:
    """Train a fresh recurrent classifier on the shared digit split.

    Args:
        opytimizer: Position rows containing SGD learning rate and momentum in that order.

    Returns:
        One minus validation accuracy after five training epochs.

    """

    n_features = 8
    n_hidden = 128
    n_classes = 10

    model = LSTM(n_features, n_hidden, n_classes)

    batch_size = 100
    epochs = 5

    learning_rate = opytimizer[0][0]
    momentum = opytimizer[1][0]

    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for _ in range(epochs):
        cost = 0.0
        num_batches = len(Y_train) // batch_size

        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += fit(model, loss, opt, X_train[:, start:end, :], Y_train[start:end])

    preds = predict(model, X_val)
    acc = np.mean(preds == Y_val)

    return 1 - acc


n_agents = 10
n_variables = 2

lower_bound = [0, 0]
upper_bound = [1, 1]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

opt = Opytimizer(space, optimizer, lstm)

opt.start(n_iterations=100)
