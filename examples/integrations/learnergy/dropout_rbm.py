import torchvision
from learnergy.models.bernoulli import DropoutRBM

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

# Creates training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())


def dropout_rbm(opytimizer):
    # Gathers params
    dropout = opytimizer[0][0]

    # Creates an RBM
    model = DropoutRBM(n_visible=784, n_hidden=128, steps=1, learning_rate=0.1,
                       momentum=0, decay=0, temperature=1, dropout=dropout, use_gpu=False)

    # Training an RBM
    error, _ = model.fit(train, batch_size=128, epochs=5)

    return error


# Number of agents and decision variables
n_agents = 5
n_variables = 1

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [0]
upper_bound = [1]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = Function(dropout_rbm)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=5)
