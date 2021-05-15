from opytimizer.optimizers.swarm import SSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'C_w': 0.1,
    'C_p': 0.4,
    'C_g': 0.9
}

# Creates a SSO optimizer
o = SSO(params=params)
