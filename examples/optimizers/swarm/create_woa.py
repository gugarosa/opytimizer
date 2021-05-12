from opytimizer.optimizers.swarm.woa import WOA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'b': 1
}

# Creates an WOA optimizer
o = WOA(params=params)
