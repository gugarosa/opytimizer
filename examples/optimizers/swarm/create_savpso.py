from opytimizer.optimizers.swarm.pso import SAVPSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'w': 0.7,
    'c1': 1.7,
    'c2': 1.7
}

# Creating an SAVPSO optimizer
o = SAVPSO(hyperparams=hyperparams)
