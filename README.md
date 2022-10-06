# Opytimizer: A Nature-Inspired Python Optimizer

[![Latest release](https://img.shields.io/github/release/gugarosa/opytimizer.svg)](https://github.com/gugarosa/opytimizer/releases)
[![DOI](http://img.shields.io/badge/DOI-10.5281/zenodo.4594294-006DB9.svg)](https://doi.org/10.5281/zenodo.4594294)
[![Build status](https://img.shields.io/travis/com/gugarosa/opytimizer/master.svg)](https://github.com/gugarosa/opytimizer/releases)
[![Open issues](https://img.shields.io/github/issues/gugarosa/opytimizer.svg)](https://github.com/gugarosa/opytimizer/issues)
[![License](https://img.shields.io/github/license/gugarosa/opytimizer.svg)](https://github.com/gugarosa/opytimizer/blob/master/LICENSE)

## Welcome to Opytimizer.
Did you ever reach a bottleneck in your computational experiments? Are you tired of selecting suitable parameters for a chosen technique? If yes, Opytimizer is the real deal! This package provides an easy-to-go implementation of meta-heuristic optimizations. From agents to search space, from internal functions to external communication, we will foster all research related to optimizing stuff.

Use Opytimizer if you need a library or wish to:
* Create your optimization algorithm;
* Design or use pre-loaded optimization tasks;
* Mix-and-match different strategies to solve your problem;
* Because it is fun to optimize things.

Read the docs at [opytimizer.readthedocs.io](https://opytimizer.readthedocs.io).

Opytimizer is compatible with: **Python 3.6+**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy if you wish to read the code and bump yourself into, follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**, call us.
5. Finally, we focus on **minimization**. Take that in mind when designing your problem.

---

## Citation

If you use Opytimizer to fulfill any of your needs, please cite us:

```BibTex
@misc{rosa2019opytimizer,
    title={Opytimizer: A Nature-Inspired Python Optimizer},
    author={Gustavo H. de Rosa, Douglas Rodrigues and João P. Papa},
    year={2019},
    eprint={1912.13002},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```

---

## Getting started: 60 seconds with Opytimizer

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, chose your subpackage, and follow the example. We have high-level examples for most tasks we could think of and amazing integrations ([Learnergy](https://github.com/gugarosa/learnergy), [NALP](https://github.com/gugarosa/NALP), [OPFython](https://github.com/gugarosa/OPFython), [PyTorch](https://github.com/pytorch/pytorch), [Scikit-Learn](https://github.com/scikit-learn/scikit-learn), [Tensorflow](https://github.com/tensorflow/tensorflow)).

Alternatively, if you wish to learn even more, please take a minute:

Opytimizer is based on the following structure, and you should pay attention to its tree:

```yaml
- opytimizer
    - core
        - agent
        - block
        - cell
        - function
        - node
        - optimizer
        - space
    - functions
        - constrained
        - multi_objective
    - math
        - distribution
        - general
        - hyper
        - random
    - optimizers
        - boolean
        - evolutionary
        - misc
        - population
        - science
        - social
        - swarm
    - spaces
        - boolean
        - graph
        - grid
        - hyper_complex
        - pareto
        - search
        - tree
    - utils
        - callback
        - constant
        - exception
        - history
        - logging
    - visualization
        - convergence
        - surface
```

### Core

Core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basis of our structure. They should provide variables and methods that will help to construct other modules.

### Functions

Instead of using raw and straightforward functions, why not try this module? Compose high-level abstract functions or even new function-based ideas in order to solve your problems. Note that for now, we will only support multi-objective function strategies.

### Math

Just because we are computing stuff does not means that we do not need math. Math is the mathematical package containing low-level math implementations. From random numbers to distribution generation, you can find your needs on this module.

### Optimizers

This is why we are called Opytimizer. This is the heart of heuristics, where you can find a large number of meta-heuristics, optimization techniques, anything that can be called an optimizer. Please take a look at the [available optimizers](https://github.com/gugarosa/opytimizer/wiki/Types-of-Optimizers).

### Spaces

One can see the space as the place that agents will update their positions and evaluate a fitness function. However, the newest approaches may consider a different type of space. Thinking about that, we are glad to support diverse space implementations.

### Utils

This is a utility package. Common things shared across the application should be implemented here. It is better to implement once and use as you wish than re-implementing the same thing repeatedly.

### Visualization

Everyone needs images and plots to help visualize what is happening, correct? This package will provide every visual-related method for you. Check a specific variable convergence, your fitness function convergence, plot benchmark function surfaces, and much more!

---

## Installation

We believe that everything has to be easy. Not tricky or daunting, Opytimizer will be the one-to-go package that you will need, from the first installation to the daily tasks implementing needs. If you may just run the following under your most preferred Python environment (raw, conda, virtualenv, whatever):

```bash
pip install opytimizer
```

Alternatively, if you prefer to install the bleeding-edge version, please clone this repository and use:

```bash
pip install -e .
```

---

## Environment configuration

Note that sometimes, there is a need for additional implementation. If needed, from here, you will be the one to know all of its details.

### Ubuntu

No specific additional commands are needed.

### Windows

No specific additional commands are needed.

### MacOS

No specific additional commands are needed.

---

## How-To-Use: Minimal Example

Take a look at a quick working example of Opytimizer. Note that we are not passing many extra arguments nor additional information to the procedure. For more complex examples, please check our `examples/` folder.

```Python
import numpy as np

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

def sphere(x):
  return np.sum(x ** 2)

n_agents = 20
n_variables = 2
lower_bound = [-10, -10]
upper_bound = [10, 10]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = Function(sphere)

opt = Opytimizer(space, optimizer, function)
opt.start(n_iterations=1000)
```

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository.

---
