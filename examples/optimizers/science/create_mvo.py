from opytimizer.optimizers.science.mvo import MVO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'WEP_min': 0.2,
    'WEP_max': 1,
    'p': 6
}

# Creating a MVO optimizer
o = MVO(params=params)
