from opytimizer.optimizers.boolean.umda import UMDA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'p_selection': 0.75,
    'lower_bound': 0.05,
    'upper_bound': 0.95
}

# Creating a UMDA optimizer
o = UMDA(params=params)
