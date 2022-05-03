from opytimizer.optimizers.evolutionary import GHS

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"HMCR": 0.7, "PAR_min": 0.0, "PAR_max": 1.0, "bw_min": 1.0, "bw_max": 10.0}

# Creates an GHS optimizer
o = GHS(params=params)
