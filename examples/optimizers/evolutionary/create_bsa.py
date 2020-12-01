from opytimizer.optimizers.evolutionary.bsa import BSA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'F': 3.0,
    'mix_rate': 1
}

# Creating a BSA optimizer
o = BSA(hyperparams=hyperparams)
