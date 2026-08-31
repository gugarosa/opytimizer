# Opytimizer Architecture

> Version 4.0.0 · Apache 2.0 · Python 3.11+

## Overview

Opytimizer minimizes a Python callable by combining three parts:

| Part | Responsibility |
|---|---|
| Space | Owns candidate agents, bounds, and the best solution |
| Optimizer | Updates candidate positions and evaluates fitness |
| Objective | Any callable accepting one position array and returning fitness |

```python
import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace


def sphere(x):
    return np.sum(x**2)


space = SearchSpace(20, 2, [-10, -10], [10, 10])
optimizer = PSO()
optimization = Opytimizer(space, optimizer, sphere)
optimization.start(n_iterations=1000)
```

Fitness is minimized throughout the library. Maximization objectives should
return the negative value being maximized.

## Runtime flow

`Opytimizer.start()` coordinates the algorithm:

1. prepare optimizer state for the selected space;
2. evaluate the initial population;
3. update positions;
4. clip positions to the space bounds;
5. evaluate candidates and retain improvements;
6. record history and dispatch callbacks;
7. repeat for the requested iterations.

The optimizer method signatures determine which runtime values are supplied to
each algorithm. This keeps the orchestrator shared while allowing algorithms to
request values such as the space, iteration, or objective.

## Package layout

### Core

* `Agent` stores one candidate position, bounds, and fitness.
* `Space` owns the population and best agent.
* `Optimizer` provides shared evaluation behavior and the update contract.
* `Node` represents expressions used by genetic-programming optimizers.

### Spaces

The standard continuous `SearchSpace` is joined by boolean, grid,
hyper-complex, Pareto, and tree spaces. Each space controls how agents are
created and initialized while retaining the same optimizer-facing population
interface.

### Optimizers

Algorithms are grouped by inspiration:

* boolean;
* evolutionary;
* miscellaneous;
* population;
* science;
* social;
* swarm.

Concrete optimizers implement their position update and any algorithm-specific
state or evaluation behavior.

### Objective helpers

Raw callables are the default API. Optional constrained and multi-objective
helpers compose callables while remaining directly callable by the optimizer.

### Utilities

History records convergence data. Callbacks provide lifecycle hooks, checkpoint
serialization, and discrete-search projection. Specialized numerical helpers
support algorithms whose operations are not direct NumPy calls.

## Persistence

Optimization state can be saved and restored with `dill`, including callbacks
and user-defined objectives. Checkpoints use the same serialization path.

## Dependencies

The runtime has two direct dependencies:

| Package | Purpose |
|---|---|
| NumPy | Arrays and numerical operations |
| dill | Optimization-state serialization |

Example integrations declare and manage their own external machine-learning
libraries.

## Development

The repository uses uv as its only project workflow:

```bash
uv sync
uv run pytest
uv build
uv run --group docs sphinx-build -b html docs docs/_build/html
```

Project metadata, dependency groups, pytest settings, and formatter settings
live in `pyproject.toml`. GitHub Actions tests Python 3.11 through 3.13 from the
committed lockfile. Sphinx generates API pages from one autosummary entry during
documentation builds.
