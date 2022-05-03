from opytimizer.optimizers.science import TEO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"c1": True, "c2": True, "pro": 0.05, "n_TM": 4}

# Creates a TEO optimizer
o = TEO(params=params)
