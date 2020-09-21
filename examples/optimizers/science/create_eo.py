from opytimizer.optimizers.science.eo import EO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {

}

# Creating an EO optimizer
o = EO(hyperparams=hyperparams)
