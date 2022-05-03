from opytimizer.optimizers.evolutionary import FOA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"life_time": 6, "area_limit": 30, "LSC": 1, "GSC": 1, "transfer_rate": 0.1}

# Creates a FOA optimizer
o = FOA(params=params)
