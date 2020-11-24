from opytimizer.optimizers.swarm.eho import EHO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'alpha': 0.5,
    'beta': 0.1,
    'n_clans': 10
}

# Creating an EHO optimizer
o = EHO(hyperparams=hyperparams)
