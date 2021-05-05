import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import optim
from torch.autograd import Variable

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.swarm.pso import PSO
from opytimizer.spaces.search import SearchSpace

# Loading digits dataset
digits = load_digits()

# Gathers samples and targets
X = digits.data
Y = digits.target

# Splitting the data
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.5, random_state=42)

# Converting from numpy array to torch tensors
X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
Y_train = torch.from_numpy(Y_train).long()


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


def logistic_regression(opytimizer):
    # Instanciating the model
    model = torch.nn.Sequential()

    # Some model parameters
    n_features = 64
    n_classes = 10

    # Adding linear layer
    model.add_module("linear", torch.nn.Linear(
        n_features, n_classes, bias=False))

    # Input variables
    batch_size = 100
    epochs = 100

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
        num_batches = len(X_train) // batch_size

        # For every batch
        for k in range(num_batches):
            # Declares initial and ending for each batch
            start, end = k * batch_size, (k + 1) * batch_size

            # Cost will be the loss accumulated from model's fitting
            cost += fit(model, loss, opt,
                        X_train[start:end], Y_train[start:end])

    # Predicting samples from evaluating set
    preds = predict(model, X_val)

    # Calculates accuracy
    acc = np.mean(preds == Y_val)

    return 1 - acc


# Creates Function's object
f = Function(pointer=logistic_regression)

# Number of agents, decision variables and iterations
n_agents = 10
n_variables = 2
n_iterations = 100

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = (0, 0)
upper_bound = (1, 1)

# Creates the SearchSpace class
s = SearchSpace(n_agents=n_agents, n_iterations=n_iterations,
                n_variables=n_variables, lower_bound=lower_bound,
                upper_bound=upper_bound)

# Parameters for the optimizer
params = {
    'w': 0.7,
    'c1': 1.7,
    'c2': 1.7
}

# Creates PSO's optimizer
p = PSO(params=params)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
history = o.start()
