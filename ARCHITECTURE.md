# Opytimizer Architecture

> Version 5.0.1 · Apache 2.0 · Python 3.11+

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

`Opytimizer` retains the supplied space and optimizer and compiles optimizer
state during construction. Repeated `start()` calls do not recompile.

`Opytimizer.start()` then coordinates each run:

1. validate the iteration budget and dispatch the task-begin hooks;
2. evaluate the initial population between evaluation hooks;
3. update positions between update hooks, then clip to bounds;
4. evaluate candidates, record history, and dispatch iteration-end hooks;
5. repeat for the requested iterations;
6. dispatch task-end hooks and record elapsed time on normal completion.

The optimizer method signatures determine which runtime values are supplied to
each algorithm. This keeps the orchestrator shared while allowing algorithms to
request values such as the space, iteration, or objective.

Iteration-local counters restart while the cumulative counter and optimizer state
continue. Callback sequences are per invocation. Exceptions propagate without
rollback or a guaranteed task-end hook. See the [usage guide](docs/usage.rst) for
ordering, history shapes, repeated runs, and resource ownership.

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

Optimization state can be saved and restored with `dill`, including the space,
optimizer, user-defined objective, history, and counters. The driver does not
retain its callback sequence; supply callbacks again when resuming. Checkpoints
use the same serialization path and do not recompile the restored optimizer.

Only load trusted checkpoints: pickle-based loading can execute code. Keep the
software environment compatible rather than treating checkpoints as a
version-independent interchange format.

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
uv run --group docs sphinx-build -b doctest docs docs/_build/doctest
```

Project metadata, dependency groups, pytest settings, and formatter settings
live in `pyproject.toml`. GitHub Actions tests Python 3.11 through 3.13 from the
committed lockfile. It also builds a wheel and checks optimization and checkpoint
round trips in an isolated environment using the lowest compatible runtime
dependencies. Interpreter-specific dependency minimums retain NumPy 1.x support
on Python 3.11 and 3.12; the lockfile does not pin library consumers.
Sphinx generates API pages from one autosummary entry during documentation builds.
Warnings-as-errors documentation builds and executable guide examples are also CI
gates. The [contributor guide](docs/development.rst) explains extension hooks,
state ownership, docstring conventions, and behavioral regression expectations.

Successful main-branch CI publishes an unreleased project version to GitHub
with wheel and source-distribution assets. Pull requests and feature branches
never publish releases. Existing releases are left unchanged, and this workflow
does not publish to PyPI.
