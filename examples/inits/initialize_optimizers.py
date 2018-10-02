from opytimizer.optimizers.pso import PSO

# One should declare a hyperparams object based
# on which algorithm that will be used
hyperparams = {
    'w': 2.5
}

# Instanciates a PSO optimizer
p = PSO(hyperparams=hyperparams)

# Prior using any Optimizer class childs, you need to build it
p.build()