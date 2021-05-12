from opytimizer.optimizers.misc.aoa import AOA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'a_min': 0.2,
    'a_max': 1.0,
    'alpha': 5,
    'mu': 0.499
}

# Creates an AOA optimizer
o = AOA(params=params)
