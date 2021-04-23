from opytimizer.optimizers.science.hgso import HGSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'n_clusters': 2,
    'l1': 0.0005,
    'l2': 100,
    'l3': 0.001,
    'alpha': 1.0,
    'beta': 1.0,
    'K': 1.0
}

# Creating an HGSO optimizer
o = HGSO(params=params)
