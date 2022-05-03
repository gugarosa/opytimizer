import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

# Setting up a random seed
torch.manual_seed(42)

# Creates X and Y data
# Note that it is important to mantain consistency during opytimizer tasks
X = torch.linspace(-1, 1, 101)
Y = 2 * X + torch.randn(X.size()) * 0.33


def fit(model, loss, opt, x, y):
    # Declares initial variables
    x = Variable(x, requires_grad=False)
    y = Variable(y, requires_grad=False)

    # Resets the gradient
    opt.zero_grad()

    # Performs the foward pass
    fw_x = model.forward(x.view(len(x), 1)).squeeze()
    output = loss.forward(fw_x, y)

    # Performs backward pass
    output.backward()

    # Updates parameters
    opt.step()

    return output.item()


def linear_regression(opytimizer):
    # Instanciating the model
    model = torch.nn.Sequential()

    # Adding linear layer
    model.add_module("linear", torch.nn.Linear(1, 1, bias=False))

    # Input variables
    batch_size = 10
    epochs = 100

    # Gathers parameters from Opytimizer
    # Pay extremely attention to their order when declaring due to their bounds
    learning_rate = opytimizer[0][0]
    momentum = opytimizer[1][0]

    # Declares the loss function
    loss = torch.nn.MSELoss(reduction="mean")

    # Declares the optimization algorithm
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Performs training loop
    for _ in range(epochs):
        # Initial cost as 0.0
        cost = 0.0

        # Calculates the number of batches
        num_batches = len(X) // batch_size

        # For every batch
        for k in range(num_batches):
            # Declares initial and ending for each batch
            start, end = k * batch_size, (k + 1) * batch_size

            # Cost will be the loss accumulated from model's fitting
            cost += fit(model, loss, opt, X[start:end], Y[start:end])

    # Calculates final cost
    final_cost = cost / num_batches

    return final_cost


# Number of agents and decision variables
n_agents = 10
n_variables = 2

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [0, 0]
upper_bound = [1, 1]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = Function(linear_regression)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=100)
