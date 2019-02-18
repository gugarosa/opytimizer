from opytimizer.optimizers.pso import PSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'w': 2.5,
    'c1': 1.5,
    'c2': 2
}

# Creating a PSO optimizer
p = PSO(hyperparams=hyperparams)
