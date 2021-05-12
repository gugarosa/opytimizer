from opytimizer.optimizers.evolutionary.bsa import BSA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'F': 3.0,
    'mix_rate': 1
}

# Creates a BSA optimizer
o = BSA(params=params)
