from opytimizer.optimizers.evolutionary import EP

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"bout_size": 0.1, "clip_ratio": 0.05}

# Creates an EP optimizer
o = EP(params=params)
