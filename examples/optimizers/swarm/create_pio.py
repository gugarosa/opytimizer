from opytimizer.optimizers.swarm.pio import PIO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'n_c1': 150,
    'n_c2': 200,
    'R': 0.2
}

# Creates an PIO optimizer
o = PIO(params=params)
