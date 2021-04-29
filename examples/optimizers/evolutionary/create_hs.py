from opytimizer.optimizers.evolutionary.hs import HS

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'HMCR': 0.7,
    'PAR': 0.7,
    'bw': 1.0
}

# Creates a HS optimizer
o = HS(params=params)
