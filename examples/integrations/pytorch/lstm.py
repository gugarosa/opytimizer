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

# Gathering samples and targets
X = digits.data
Y = digits.target

# Splitting the data
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.5, random_state=42)

# Reshaping the data
X_train = X_train.reshape(-1, 8, 8)
X_val = X_val.reshape(-1, 8, 8)

# Converting to sequence shape
X_train = np.swapaxes(X_train, 0, 1)
X_val = np.swapaxes(X_val, 0, 1)

# Converting from numpy array to torch tensors
X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
Y_train = torch.from_numpy(Y_train).long()


class LSTM(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_classes):
        # Overriding initial class
        super(LSTM, self).__init__()

        # Saving number of hidden units as a property
        self.n_hidden = n_hidden

        # Creating LSTM cell
        self.lstm = torch.nn.LSTM(n_features, n_hidden)

        # Creating linear layer
        self.linear = torch.nn.Linear(n_hidden, n_classes, bias=False)

    def forward(self, x):
        # Gathering batch size
        batch_size = x.size()[1]

        # Variable to hold hidden state
        h0 = Variable(torch.zeros(
            [1, batch_size, self.n_hidden]), requires_grad=False)

        # Variable to hold cell state
        c0 = Variable(torch.zeros(
            [1, batch_size, self.n_hidden]), requires_grad=False)

        # Performing forward pass
        fx, _ = self.lstm.forward(x, (h0, c0))

        return self.linear.forward(fx[-1])


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

    # Gathering parameters from Opytimizer
    # Pay extremely attention to their order when declaring due to their bounds
    learning_rate = opytimizer[0][0]
    momentum = opytimizer[1][0]

    # Declaring the loss function
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    # Declaring the optimization algorithm
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Performing training loop
    for _ in range(epochs):
        # Initial cost as 0.0
        cost = 0.0

        # Calculating the number of batches
        num_batches = len(Y_train) // batch_size

        # For every batch
        for k in range(num_batches):
            # Declaring initial and ending for each batch
            start, end = k * batch_size, (k + 1) * batch_size

            # Cost will be the loss accumulated from model's fitting
            cost += fit(model, loss, opt,
                        X_train[:, start:end, :], Y_train[start:end])

    # Predicting samples from evaluating set
    preds = predict(model, X_val)

    # Calculating accuracy
    acc = np.mean(preds == Y_val)

    return 1 - acc


# Creating Function's object
f = Function(pointer=lstm)

# Number of agents
n_agents = 10

# Number of decision variables
n_variables = 2

# Number of running iterations
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
