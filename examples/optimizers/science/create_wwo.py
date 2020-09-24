from opytimizer.optimizers.science.wwo import WWO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'h_max': 5,
    'alpha': 1.001,
    'beta': 0.001,
    'k_max': 1
}

# Creating a WWO optimizer
o = WWO(hyperparams=hyperparams)
