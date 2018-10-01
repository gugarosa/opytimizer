from opytimizer.optimizers.pso import PSO

# One should declare a hyperparams object based
# on which algorithm that will be used
hyperparams = {
    'w': 2.5
}

# Instanciates a PSO optimizer
p = PSO(hyperparams=hyperparams)

# Prints important information about it
print(p.hyperparams)