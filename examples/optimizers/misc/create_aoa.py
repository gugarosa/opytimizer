from opytimizer.optimizers.misc.aoa import AOA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'a_min': 0.2,
    'a_max': 1.0,
    'alpha': 5,
    'mu': 0.499
}

# Creating an AOA optimizer
o = AOA(hyperparams=hyperparams)
