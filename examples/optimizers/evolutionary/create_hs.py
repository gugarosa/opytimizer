from opytimizer.optimizers.hs import HS

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'HMCR': 0.7,
    'PAR': 0.7,
    'bw': 1.0
}

# Creating a HS optimizer
o = HS(hyperparams=hyperparams)
