from opytimizer.optimizers.evolutionary import SGHS

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    "HMCR": 0.7,
    "PAR": 0.7,
    "LP": 100,
    "HMCRm": 0.98,
    "PARm": 0.9,
    "bw_min": 1.0,
    "bw_max": 10.0,
}

# Creates a SGHS optimizer
o = SGHS(params=params)
