from opytimizer.optimizers.science.wwo import WWO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'h_max': 5,
    'alpha': 1.001,
    'beta': 0.001,
    'k_max': 1
}

# Creates a WWO optimizer
o = WWO(params=params)
