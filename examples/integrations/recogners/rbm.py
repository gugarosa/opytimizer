import torchvision
from recogners.models.rbm import RBM
from torch.utils.data import DataLoader

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.pso import PSO
from opytimizer.spaces.search import SearchSpace

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

# Creating training and testing batches
train_batches = DataLoader(train, batch_size=128, shuffle=True, num_workers=1)


def rbm(opytimizer):
    # Gathering hyperparams
    lr = opytimizer[0][0]
    momentum = opytimizer[1][0]
    decay = opytimizer[2][0]

    # Creating an RBM
    model = RBM(n_visible=784, n_hidden=128, steps=1, learning_rate=lr,
                momentum=momentum, decay=decay, temperature=1)

    # Training an RBM
    error, pl = model.fit(train_batches, epochs=5)

    return error


# Creating Function's object
f = Function(pointer=rbm)

# Number of agents
n_agents = 10

# Number of decision variables
n_variables = 3

# Number of running iterations
n_iterations = 10

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
