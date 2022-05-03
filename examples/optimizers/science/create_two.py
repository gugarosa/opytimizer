from opytimizer.optimizers.science import TWO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"mu_s": 1, "mu_k": 1, "delta_t": 1, "alpha": 0.9, "beta": 0.05}

# Creates a TWO optimizer
o = TWO(params=params)
