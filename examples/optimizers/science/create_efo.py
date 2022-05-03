from opytimizer.optimizers.science import EFO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    "positive_field": 0.1,
    "negative_field": 0.5,
    "ps_ratio": 0.1,
    "r_ratio": 0.4,
}

# Creates an EFO optimizer
o = EFO(params=params)
