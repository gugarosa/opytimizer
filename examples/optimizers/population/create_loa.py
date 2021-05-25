from opytimizer.optimizers.population import LOA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'N': 0.2,
    'P': 4,
    'S': 0.8,
    'R': 0.2,
    'I': 0.4,
    'Ma': 0.3,
    'Mu': 0.2
}

# Creates an LOA optimizer
o = LOA(params=params)
