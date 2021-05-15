from opytimizer.optimizers.evolutionary import NGHS

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'pm': 0.1
}

# Creates a NGHS optimizer
o = NGHS(params=params)
