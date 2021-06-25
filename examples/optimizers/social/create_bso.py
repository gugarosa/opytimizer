from opytimizer.optimizers.social import BSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'k': 5,
    'p_single_cluster': 0.3,
    'p_single_idea': 0.4,
    'p_double_idea': 0.3
}

# Creates an BSO optimizer
o = BSO(params=params)
