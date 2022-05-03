from opytimizer.optimizers.social import SSD

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"c": 2.0, "decay": 0.99}

# Creates an SSD optimizer
o = SSD(params=params)
