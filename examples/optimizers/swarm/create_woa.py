from opytimizer.optimizers.swarm.woa import WOA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'b': 1
}

# Creating an WOA optimizer
o = WOA(hyperparams=hyperparams)
