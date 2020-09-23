from opytimizer.optimizers.science.eo import EO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'a1': 2,
    'a2': 1,
    'GP': 0.5,
    'V': 1
}

# Creating an EO optimizer
o = EO(hyperparams=hyperparams)
