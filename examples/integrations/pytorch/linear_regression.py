import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.swarm.pso import PSO
from opytimizer.spaces.search import SearchSpace

# Setting up a random seed
torch.manual_seed(42)

# Creating X and Y data
# Note that it is important to mantain consistency during opytimizer tasks
X = torch.linspace(-1, 1, 101)
Y = 2 * X + torch.randn(X.size()) * 0.33

def fit(model, loss, opt, x, y):
    # Declaring initial variables
    x = Variable(x, requires_grad=False)
    y = Variable(y, requires_grad=False)

    # Resetting the gradient
    opt.zero_grad()

    # Performing the foward pass
    fw_x = model.forward(x.view(len(x), 1)).squeeze()
    output = loss.forward(fw_x, y)

    # Performing backward pass
    output.backward()

    # Updating parameters
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

    # Gathering parameters from Opytimizer
    # Pay extremely attention to their order when declaring due to their bounds
    learning_rate = opytimizer[0][0]
    momentum = opytimizer[1][0]

    # Declaring the loss function
    loss = torch.nn.MSELoss(reduction='mean')

    # Declaring the optimization algorithm
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Performing training loop
    for _ in range(epochs):
        # Initial cost as 0.0
        cost = 0.0

        # Calculating the number of batches
        num_batches = len(X) // batch_size

        # For every batch
        for k in range(num_batches):
            # Declaring initial and ending for each batch
            start, end = k * batch_size, (k + 1) * batch_size

            # Cost will be the loss accumulated from model's fitting
            cost += fit(model, loss, opt, X[start:end], Y[start:end])

    # Calculating final cost
    final_cost = cost / num_batches

    return final_cost

# Creating Function's object
f = Function(pointer=linear_regression)

# Number of agents, decision variables and iterations
n_agents = 10
n_variables = 2
n_iterations = 100

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = (0, 0)
upper_bound = (1, 1)

# Creating the SearchSpace class
s = SearchSpace(n_agents=n_agents, n_iterations=n_iterations,
                n_variables=n_variables, lower_bound=lower_bound,
                upper_bound=upper_bound)

# Hyperparameters for the optimizer
hyperparams = {
    'w': 0.7,
    'c1': 1.7,
    'c2': 1.7
}

# Creating PSO's optimizer
p = PSO(hyperparams=hyperparams)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
history = o.start()
