from opytimizer.optimizers.population.ao import AO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'alpha': 0.1
    'delta': 0.1,
    'n_cycles': 10,
    'U': 0.00565,
    'w': 0.005
}

# Creating an AO optimizer
o = AO(hyperparams=hyperparams)
