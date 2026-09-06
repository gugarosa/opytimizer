# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Plot recorded fitness directly with the optional Matplotlib dependency.

Install Matplotlib separately to run this recipe. It is not an Opytimizer runtime
dependency. Set ``MPLBACKEND=Agg`` when running without an interactive display.

"""

import matplotlib.pyplot as plt
import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace


def sphere(x: np.ndarray) -> float:
    """Evaluate the sphere objective.

    Args:
        x: Candidate position array.

    Returns:
        Sum of squared decision variables.

    """

    return np.sum(x**2)


np.random.seed(0)
space = SearchSpace(20, 2, [-5, -5], [5, 5])
opt = Opytimizer(space, PSO(), sphere, save_agents=True)
opt.start(n_iterations=100)

_, best_fitness = opt.history.get_convergence("best_agent")
_, agent_fitness = opt.history.get_convergence("agents", index=0)
iterations = np.arange(1, len(best_fitness) + 1)

figure, axes = plt.subplots()
axes.plot(iterations, np.asarray(best_fitness, dtype=float), label="Best fitness")
# PSO stores each candidate's personal-best fitness, not its current-position fitness
axes.plot(iterations, np.asarray(agent_fitness, dtype=float), label="Agent 0 recorded fitness")
axes.set(xlabel="Iteration", ylabel="Fitness", title="Sphere optimization convergence")
axes.legend()
figure.tight_layout()
plt.show()
