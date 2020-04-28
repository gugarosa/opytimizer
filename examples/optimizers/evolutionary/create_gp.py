from opytimizer.optimizers.evolutionary.gp import GP

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'p_reproduction': 0.25,
    'p_mutation': 0.1,
    'p_crossover': 0.2,
    'prunning_ratio': 0.0
}

# Creating a GP optimizer
o = GP(hyperparams=hyperparams)
