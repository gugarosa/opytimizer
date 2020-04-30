from opytimizer.optimizers.evolutionary.ga import GA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'p_selection': 0.75,
    'p_mutation': 0.25,
    'p_crossover': 0.5,
}

# Creating a GA optimizer
o = GA(hyperparams=hyperparams)
