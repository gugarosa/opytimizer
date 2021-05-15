from opytimizer.optimizers.misc import HC

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'r_mean': 0,
    'r_var': 0.1
}

# Creates a HC optimizer
o = HC(params=params)
