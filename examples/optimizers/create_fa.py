from opytimizer.optimizers.fa import FA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'alpha': 0.5,
    'beta': 0.2,
    'gamma': 1.0
}

# Creating a FA optimizer
o = FA(hyperparams=hyperparams)
