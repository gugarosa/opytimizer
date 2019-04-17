from opytimizer.optimizers.ihs import IHS

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'HMCR': 0.7,
    'PAR_min': 0,
    'PAR_max': 1,
    'bw_min': 1,
    'bw_max': 10
}

# Creating an IHS optimizer
o = IHS(hyperparams=hyperparams)
