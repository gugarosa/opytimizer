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

# Loads digits dataset
digits = load_digits()

# Gathers samples and targets
X = digits.data
Y = digits.target

# Splits the data
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.5, random_state=42)

# Reshapes the data
X_train = X_train.reshape(-1, 8, 8)
X_val = X_val.reshape(-1, 8, 8)

# Converts to sequence shape
X_train = np.swapaxes(X_train, 0, 1)
X_val = np.swapaxes(X_val, 0, 1)

# Converts from numpy array to torch tensors
X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
Y_train = torch.from_numpy(Y_train).long()


class LSTM(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_classes):
        # Overriding initial class
        super(LSTM, self).__init__()

        # Saving number of hidden units as a property
        self.n_hidden = n_hidden

        # Creates LSTM cell
        self.lstm = torch.nn.LSTM(n_features, n_hidden)

        # Creates linear layer
        self.linear = torch.nn.Linear(n_hidden, n_classes, bias=False)

    def forward(self, x):
        # Gathers batch size
        batch_size = x.size()[1]

        # Variable to hold hidden state
        h0 = Variable(torch.zeros(
            [1, batch_size, self.n_hidden]), requires_grad=False)

        # Variable to hold cell state
        c0 = Variable(torch.zeros(
            [1, batch_size, self.n_hidden]), requires_grad=False)

        # Performs forward pass
        fx, _ = self.lstm.forward(x, (h0, c0))

        return self.linear.forward(fx[-1])


def fit(model, loss, opt, x, y):
    # Declares initial variables
    x = Variable(x, requires_grad=False)
    y = Variable(y, requires_grad=False)

    # Resets the gradient
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

    # Gets the index of the prediction
    y_val = output.data.numpy().argmax(axis=1)

    return y_val


def lstm(opytimizer):
    # Some model parameters
    n_features = 8
    n_hidden = 128
    n_classes = 10

    # Instanciating the model
    model = LSTM(n_features, n_hidden, n_classes)

    # Input variables
    batch_size = 100
    epochs = 5

    # Gathers parameters from Opytimizer
    # Pay extremely attention to their order when declaring due to their bounds
    learning_rate = opytimizer[0][0]
    momentum = opytimizer[1][0]

    # Declares the loss function
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    # Declares the optimization algorithm
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Performs training loop
    for _ in range(epochs):
        # Initial cost as 0.0
        cost = 0.0

        # Calculates the number of batches
        num_batches = len(Y_train) // batch_size

        # For every batch
        for k in range(num_batches):
            # Declares initial and ending for each batch
            start, end = k * batch_size, (k + 1) * batch_size

            # Cost will be the loss accumulated from model's fitting
            cost += fit(model, loss, opt,
                        X_train[:, start:end, :], Y_train[start:end])

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
function = Function(lstm)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=100)
