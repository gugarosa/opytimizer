from opytimizer.optimizers.social.ssd import SSD

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'c': 2.0,
    'decay': 0.99
}

# Creating an SSD optimizer
o = SSD(hyperparams=hyperparams)
