from opytimizer.optimizers.gco import GCO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'CR': 0.7,
    'F': 1.25
}

# Creating a GCO optimizer
o = GCO(hyperparams=hyperparams)
