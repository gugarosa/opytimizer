import gc

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.swarm.pso import PSO
from opytimizer.spaces.search import SearchSpace

# Loading CIFAR-10 data
(X_train, Y_train), (X_val, Y_val) = datasets.cifar10.load_data()

# Normalizing inputs between 0 and 1
X_train, X_val = X_train / 255.0, X_val / 255.0


def cnn(opytimizer):
    # Gathers parameters from Opytimizer
    # Pay extremely attention to their order when declaring due to their bounds
    learning_rate = opytimizer[0][0]
    beta_1 = opytimizer[1][0]

    # Instanciating model
    model = models.Sequential()

    # Adding layers to the model itself
    model.add(layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compiling the model
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Fitting the model
    history = model.fit(X_train, Y_train, epochs=3, validation_data=(X_val, Y_val))

    # Gathers validation accuracy
    val_acc = history.history['val_accuracy'][-1]

    # Cleaning up memory
    del history
    del model

    # Calling the garbage collector
    gc.collect()

    return 1 - val_acc


# Creating Function's object
f = Function(pointer=cnn)

# Number of agents, decision variables and iterations
n_agents = 5
n_variables = 2
n_iterations = 3

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = (0, 0)
upper_bound = (0.001, 1)

# Creating the SearchSpace class
s = SearchSpace(n_agents=n_agents, n_iterations=n_iterations,
                n_variables=n_variables, lower_bound=lower_bound,
                upper_bound=upper_bound)

# Parameters for the optimizer
params = {
    'w': 0.7,
    'c1': 1.7,
    'c2': 1.7
}

# Creating PSO's optimizer
p = PSO(params=params)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
history = o.start()
