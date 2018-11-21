from opytimizer.optimizers.pso import PSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'w': 2.5
}

# Creating a PSO optimizer
p = PSO(hyperparams=hyperparams)
