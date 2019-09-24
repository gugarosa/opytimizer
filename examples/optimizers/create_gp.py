from opytimizer.optimizers.gp import GP

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'reproduction': 0.3,
    'mutation': 0.4,
    'crossover': 0.4
}

# Creating a GP optimizer
o = GP(hyperparams=hyperparams)
