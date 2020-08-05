from opytimizer.optimizers.evolutionary.hs import SGHS

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'HMCRm': 0.98,
    'PARm': 0.9,
    'bw_min': 1.0,
    'bw_max': 10.0
}

# Creating a SGHS optimizer
o = SGHS(hyperparams=hyperparams)
