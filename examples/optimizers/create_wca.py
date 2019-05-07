from opytimizer.optimizers.wca import WCA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'nsr': 2,
    'd_max': 0.1
}

# Creating a WCA optimizer
o = WCA(hyperparams=hyperparams)
