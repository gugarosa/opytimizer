import numpy as np
import torch
from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.pso import PSO
from opytimizer.spaces.search import SearchSpace
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import optim
from torch.autograd import Variable

# Loading digits dataset
digits = load_digits()

# Gathering samples and targets
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
    # Declaring initial variables
    x = Variable(x, requires_grad=False)
    y = Variable(y, requires_grad=False)

    # Resetting the gradient
    opt.zero_grad()

    # Performing the foward pass
    fw_x = model.forward(x)
    output = loss.forward(fw_x, y)

    # Performing backward pass
    output.backward()

    # Updating parameters
    opt.step()

    return output.item()


def predict(model, x_val):
    # Declaring validation variable
    x = Variable(x_val, requires_grad=False)

    # Performing backward pass with this variable
    output = model.forward(x)

    # Getting the index of the prediction
    y_val = output.data.numpy().argmax(axis=1)

    return y_val


def enhanced_neural_network(opytimizer):
    # Instanciating the model
    model = torch.nn.Sequential()

    # Some model parameters
    n_features = 64
    n_hidden = 128
    n_classes = 10

    # Adding first linear layer
    model.add_module("linear_1", torch.nn.Linear(
        n_features, n_hidden, bias=False))

    # Activated by ReLU
    model.add_module("relu_1", torch.nn.ReLU())

    # Simple dropout layer
    model.add_module("dropout_1", torch.nn.Dropout(0.2))

    # Adding secondary linear layer
    model.add_module("linear_2", torch.nn.Linear(
        n_hidden, n_hidden, bias=False))

    # Activated by ReLU
    model.add_module("relu_2", torch.nn.ReLU())

    # Yet another simple dropout layer
    model.add_module("dropout_2", torch.nn.Dropout(0.2))

    # Final linear layer
    model.add_module("linear_3", torch.nn.Linear(
        n_hidden, n_classes, bias=False))

    # Input variables
    batch_size = 100
    epochs = 100

    # Gathering parameters from Opytimizer
    # Pay extremely attention to their order when declaring due to their bounds
    learning_rate = opytimizer[0][0]
    eps = opytimizer[1][0]
    weight_decay = opytimizer[2][0]

    # Declaring the loss function
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    # Declaring the optimization algorithm
    opt = optim.Adam(model.parameters(), lr=learning_rate,
                     eps=eps, weight_decay=weight_decay)

    # Performing training loop
    for _ in range(epochs):
        # Initial cost as 0.0
        cost = 0.0

        # Calculating the number of batches
        num_batches = len(X_train) // batch_size

        # For every batch
        for k in range(num_batches):
            # Declaring initial and ending for each batch
            start, end = k * batch_size, (k + 1) * batch_size

            # Cost will be the loss accumulated from model's fitting
            cost += fit(model, loss, opt,
                        X_train[start:end], Y_train[start:end])

    # Predicting samples from evaluating set
    preds = predict(model, X_val)

    # Calculating accuracy
    acc = np.mean(preds == Y_val)

    return 1 - acc


# Creating Function's object
f = Function(pointer=enhanced_neural_network)

# Number of agents
n_agents = 10

# Number of decision variables
n_variables = 3

# Number of running iterations
n_iterations = 100

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = [0, 0, 0]
upper_bound = [1, 1, 1]

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
