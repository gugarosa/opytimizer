from opytimizer.optimizers.population.coa import COA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'n_p': 2
}

# Creating a COA optimizer
o = COA(hyperparams=hyperparams)
