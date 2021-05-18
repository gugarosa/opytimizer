from opytimizer.optimizers.social import ISA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'w': 0.7,
    'tau': 0.3
}

# Creates an ISA optimizer
o = ISA(params=params)
