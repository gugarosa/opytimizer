from opytimizer.optimizers.science import LSA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"max_time": 10, "E": 2.05, "p_fork": 0.01}

# Creates an LSA optimizer
o = LSA(params=params)
