import torchvision
from learnergy.models.binary import RBM

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.swarm.pso import PSO
from opytimizer.spaces.search import SearchSpace

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())


def rbm(opytimizer):
    # Gathering params
    lr = opytimizer[0][0]
    momentum = opytimizer[1][0]
    decay = opytimizer[2][0]

    # Creating an RBM
    model = RBM(n_visible=784, n_hidden=128, steps=1, learning_rate=lr,
                momentum=momentum, decay=decay, temperature=1, use_gpu=False)

    # Training an RBM
    error, _ = model.fit(train, batch_size=128, epochs=5)

    return error


# Creating Function's object
f = Function(pointer=rbm)

# Number of agents, decision variables and iterations
n_agents = 10
n_variables = 3
n_iterations = 10

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = (0, 0, 0)
upper_bound = (1, 1, 1)

# Creating the SearchSpace class
s = SearchSpace(n_agents=n_agents, n_iterations=n_iterations,
                n_variables=n_variables, lower_bound=lower_bound,
                upper_bound=upper_bound)

# Parameters for the optimizer
params = {
    'w': 0.7,
    'c1': 1.7,
    'c2': 1.7
}

# Creating PSO's optimizer
p = PSO(params=params)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
history = o.start()
