import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import optim
from torch.autograd import Variable

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

# Loading digits dataset
digits = load_digits()

# Gathers samples and targets
X = digits.data
Y = digits.target

# Splitting the data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.5, random_state=42)

# Reshaping the data
X_train = X_train.reshape(-1, 1, 8, 8)
X_val = X_val.reshape(-1, 1, 8, 8)

# Converting from numpy array to torch tensors
X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
Y_train = torch.from_numpy(Y_train).long()


class CNN(torch.nn.Module):
    def __init__(self, n_classes):
        # Overriding initial class
        super(CNN, self).__init__()

        # Creates sequential model for convolutional part
        self.conv = torch.nn.Sequential()

        # First convolutional block
        # Pay extremely attention when matching tensor dimensions
        self.conv.add_module("conv_1", torch.nn.Conv2d(1, 4, kernel_size=2))
        self.conv.add_module("dropout_1", torch.nn.Dropout())
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_1", torch.nn.ReLU())

        # Second convolutional block
        # Same apply for this step
        self.conv.add_module("conv_2", torch.nn.Conv2d(4, 8, kernel_size=2))
        self.conv.add_module("dropout_2", torch.nn.Dropout())
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_2", torch.nn.ReLU())

        # Creates sequential model for fully connected part
        self.fc = torch.nn.Sequential()

        # Fully connected block
        self.fc.add_module("fc1", torch.nn.Linear(8, 32))
        self.fc.add_module("relu_3", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout())
        self.fc.add_module("fc2", torch.nn.Linear(32, n_classes))

    def forward(self, x):
        # Performs first block forward step
        x = self.conv.forward(x)

        # Flattening tensor
        x = x.view(-1, 8)
        return self.fc.forward(x)


def fit(model, loss, opt, x, y):
    # Declares initial variables
    x = Variable(x, requires_grad=False)
    y = Variable(y, requires_grad=False)

    # Resetting the gradient
    opt.zero_grad()

    # Performs the foward pass
    fw_x = model.forward(x)
    output = loss.forward(fw_x, y)

    # Performs backward pass
    output.backward()

    # Updates parameters
    opt.step()

    return output.item()


def predict(model, x_val):
    # Declares validation variable
    x = Variable(x_val, requires_grad=False)

    # Performs backward pass with this variable
    output = model.forward(x)

    # Getting the index of the prediction
    y_val = output.data.numpy().argmax(axis=1)

    return y_val


def cnn(opytimizer):
    # Some model parameters
    n_classes = 10

    # Instanciating the model
    model = CNN(n_classes=n_classes)

    # Input variables
    batch_size = 100
    epochs = 50

    # Gathers parameters from Opytimizer
    # Pay extremely attention to their order when declaring due to their bounds
    learning_rate = opytimizer[0][0]
    momentum = opytimizer[1][0]

    # Declares the loss function
    loss = torch.nn.CrossEntropyLoss(reduction="mean")

    # Declares the optimization algorithm
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Performs training loop
    for _ in range(epochs):
        # Initial cost as 0.0
        cost = 0.0

        # Calculates the number of batches
        num_batches = len(X_train) // batch_size

        # For every batch
        for k in range(num_batches):
            # Declares initial and ending for each batch
            start, end = k * batch_size, (k + 1) * batch_size

            # Cost will be the loss accumulated from model's fitting
            cost += fit(model, loss, opt, X_train[start:end], Y_train[start:end])

    # Predicting samples from evaluating set
    preds = predict(model, X_val)

    # Calculates accuracy
    acc = np.mean(preds == Y_val)

    return 1 - acc


# Number of agents and decision variables
n_agents = 10
n_variables = 2

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [0, 0]
upper_bound = [1, 1]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = Function(cnn)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=100)
