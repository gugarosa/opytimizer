from opytimizer.optimizers.swarm.pio import PIO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'n_c1': 150,
    'n_c2': 200,
    'R': 0.2
}

# Creating an PIO optimizer
o = PIO(hyperparams=hyperparams)
