from opytimizer.optimizers.science.aso import ASO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'alpha': 50.0,
    'beta': 0.2
}

# Creating an ASO optimizer
o = ASO(params=params)
