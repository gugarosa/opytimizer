from opytimizer.optimizers.social import BSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'm': 5,
    'p_replacement_cluster': 0.2,
    'p_single_cluster': 0.8,
    'p_single_best': 0.4,
    'p_double_best': 0.5,
    'k': 20
}

# Creates an BSO optimizer
o = BSO(params=params)
