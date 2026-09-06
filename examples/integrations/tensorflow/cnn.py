# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import gc

import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

(X_train, Y_train), (X_val, Y_val) = datasets.cifar10.load_data()

X_train, X_val = X_train / 255.0, X_val / 255.0


def cnn(opytimizer: np.ndarray) -> float:
    """Train a fresh CNN on CIFAR-10 and release its local training references.

    Args:
        opytimizer: Position rows containing Adam learning rate and beta_1 in that order.

    Returns:
        One minus validation accuracy after three epochs on the shared dataset.

    """

    learning_rate = opytimizer[0][0]
    beta_1 = opytimizer[1][0]

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(X_train, Y_train, epochs=3, validation_data=(X_val, Y_val))

    val_acc = history.history["val_accuracy"][-1]

    # Repeated objective evaluations otherwise retain large training allocations longer
    del history
    del model

    gc.collect()

    return 1 - val_acc


n_agents = 5
n_variables = 2

lower_bound = [0, 0]
upper_bound = [0.001, 1]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

opt = Opytimizer(space, optimizer, cnn)

opt.start(n_iterations=3)
