from opytimizer.optimizers.hc import HC

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'r_mean': 0,
    'r_var': 0.1
}

# Creating a HC optimizer
o = HC(hyperparams=hyperparams)
