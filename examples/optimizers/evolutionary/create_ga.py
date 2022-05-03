from opytimizer.optimizers.evolutionary import GA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    "p_selection": 0.75,
    "p_mutation": 0.25,
    "p_crossover": 0.5,
}

# Creates a GA optimizer
o = GA(params=params)
