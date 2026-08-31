# Opytimizer: A Nature-Inspired Python Optimizer

[![Latest release](https://img.shields.io/github/release/gugarosa/opytimizer.svg)](https://github.com/gugarosa/opytimizer/releases)
[![DOI](http://img.shields.io/badge/DOI-10.5281/zenodo.4594294-006DB9.svg)](https://doi.org/10.5281/zenodo.4594294)
[![CI](https://github.com/gugarosa/opytimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/gugarosa/opytimizer/actions/workflows/ci.yml)
[![Open issues](https://img.shields.io/github/issues/gugarosa/opytimizer.svg)](https://github.com/gugarosa/opytimizer/issues)
[![License](https://img.shields.io/github/license/gugarosa/opytimizer.svg)](https://github.com/gugarosa/opytimizer/blob/master/LICENSE)

Opytimizer provides reusable nature-inspired optimization algorithms. Agents
explore a search space while an optimizer minimizes a Python callable.

Use Opytimizer to:

* apply one of the included optimization algorithms;
* create a new optimizer;
* combine optimizers, spaces, callbacks, and objective callables;
* tune external machine-learning models.

Opytimizer requires Python 3.11 or newer and focuses on minimization. Negate an
objective when solving a maximization problem.

## Installation

Add Opytimizer to a project with uv:

```bash
uv add opytimizer
```

For development:

```bash
uv sync
uv run pytest
uv build
```

Build the documentation with:

```bash
uv run --group docs sphinx-build -b html docs docs/_build/html
```

## Minimal example

```python
import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace


def sphere(x):
    return np.sum(x**2)


space = SearchSpace(
    n_agents=20,
    n_variables=2,
    lower_bound=[-10, -10],
    upper_bound=[10, 10],
)
optimizer = PSO()

opt = Opytimizer(space, optimizer, sphere)
opt.start(n_iterations=1000)
```

More complete applications and integrations are available under `examples/`.

## Package overview

* `core`: agents, spaces, optimizers, and genetic-programming nodes;
* `optimizers`: boolean, evolutionary, miscellaneous, population, science,
  social, and swarm algorithms;
* `spaces`: boolean, grid, hyper-complex, Pareto, search, and tree spaces;
* `functions`: constrained and multi-objective callable composition;
* `math`: specialized numerical helpers used by algorithms;
* `utils`: callbacks, constants, and optimization history.

## Citation

If you use Opytimizer, please cite:

```bibtex
@misc{rosa2019opytimizer,
    title={Opytimizer: A Nature-Inspired Python Optimizer},
    author={Gustavo H. de Rosa, Douglas Rodrigues and João P. Papa},
    year={2019},
    eprint={1912.13002},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```

## Support

Please use the repository issue tracker to report bugs or request features.
