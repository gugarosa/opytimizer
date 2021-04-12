import torchvision
from learnergy.models.binary import DropoutRBM

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.swarm.pso import PSO
from opytimizer.spaces.search import SearchSpace

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())


def dropout_rbm(opytimizer):
    # Gathering hyperparams
    dropout = opytimizer[0][0]

    # Creating an RBM
    model = DropoutRBM(n_visible=784, n_hidden=128, steps=1, learning_rate=0.1,
                       momentum=0, decay=0, temperature=1, dropout=dropout, use_gpu=False)

    # Training an RBM
    error, _ = model.fit(train, batch_size=128, epochs=5)

    return error


# Creating Function's object
f = Function(pointer=dropout_rbm)

# Number of agents
n_agents = 5

# Number of decision variables
n_variables = 1

# Number of running iterations
n_iterations = 5

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = (0,)
upper_bound = (1,)

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
