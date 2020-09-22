from opytimizer.optimizers.swarm.sso import SSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'C_w': 0.1,
    'C_p': 0.4,
    'C_g': 0.9
}

# Creating a SSO optimizer
o = SSO(hyperparams=hyperparams)
