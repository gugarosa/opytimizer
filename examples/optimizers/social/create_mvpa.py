from opytimizer.optimizers.social import MVPA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"n_teams": 4}

# Creates a MVPA optimizer
o = MVPA(params=params)
