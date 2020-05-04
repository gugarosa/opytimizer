from opytimizer.optimizers.science.wdo import WDO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'v_max': 0.3,
    'alpha': 0.8,
    'g': 0.6,
    'c': 1,
    'RT': 1.5
}

# Creating a WDO optimizer
o = WDO(hyperparams=hyperparams)
