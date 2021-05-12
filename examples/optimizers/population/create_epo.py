from opytimizer.optimizers.population.epo import EPO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'f': 2.0,
    'l': 1.5
}

# Creates an EPO optimizer
o = EPO(params=params)
