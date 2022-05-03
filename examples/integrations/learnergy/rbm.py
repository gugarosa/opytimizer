import torchvision
from learnergy.models.bernoulli import RBM

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

# Creates training and testing dataset
train = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)


def rbm(opytimizer):
    # Gathers params
    lr = opytimizer[0][0]
    momentum = opytimizer[1][0]
    decay = opytimizer[2][0]

    # Creates an RBM
    model = RBM(
        n_visible=784,
        n_hidden=128,
        steps=1,
        learning_rate=lr,
        momentum=momentum,
        decay=decay,
        temperature=1,
        use_gpu=False,
    )

    # Training an RBM
    error, _ = model.fit(train, batch_size=128, epochs=5)

    return error


# Number of agents and decision variables
n_agents = 10
n_variables = 3

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [0, 0, 0]
upper_bound = [1, 1, 1]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = Function(rbm)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=10)
