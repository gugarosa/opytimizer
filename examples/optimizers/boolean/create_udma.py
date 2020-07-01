from opytimizer.optimizers.boolean.udma import UDMA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'p_selection': 0.75,
    'lower_bound': 0.05,
    'upper_bound': 0.95
}

# Creating a UDMA optimizer
o = UDMA(hyperparams=hyperparams)
