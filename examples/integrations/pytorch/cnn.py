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

# Convolution expects a channel axis before each 8-by-8 image
X_train = X_train.reshape(-1, 1, 8, 8)
X_val = X_val.reshape(-1, 1, 8, 8)

X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
Y_train = torch.from_numpy(Y_train).long()


class CNN(torch.nn.Module):
    """Classify 8-by-8 grayscale images with two convolutional blocks.

    """

    def __init__(self, n_classes: int) -> None:
        """Allocate convolutional and fully connected classification layers.

        Args:
            n_classes: Number of output classes.

        """

        super(CNN, self).__init__()

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(1, 4, kernel_size=2))
        self.conv.add_module("dropout_1", torch.nn.Dropout())
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_1", torch.nn.ReLU())

        self.conv.add_module("conv_2", torch.nn.Conv2d(4, 8, kernel_size=2))
        self.conv.add_module("dropout_2", torch.nn.Dropout())
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_2", torch.nn.ReLU())

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(8, 32))
        self.fc.add_module("relu_3", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout())
        self.fc.add_module("fc2", torch.nn.Linear(32, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv.forward(x)

        # The convolutional stack leaves eight features per image
        x = x.view(-1, 8)
        return self.fc.forward(x)


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


def cnn(opytimizer: np.ndarray) -> float:
    """Train a fresh convolutional classifier on the shared digit split.

    Args:
        opytimizer: Position rows containing SGD learning rate and momentum in that order.

    Returns:
        One minus validation accuracy after fifty training epochs.

    """

    n_classes = 10
    model = CNN(n_classes=n_classes)

    batch_size = 100
    epochs = 50

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

opt = Opytimizer(space, optimizer, cnn)

opt.start(n_iterations=100)
