# Opytimizer: A Meta-Inspired Python Optimizer

[![Latest release](https://img.shields.io/github/release/gugarosa/opytimizer.svg)](https://github.com/gugarosa/opytimizer/releases)
[![Open issues](https://img.shields.io/github/issues/gugarosa/opytimizer.svg)](https://github.com/gugarosa/opytimizer/issues)
[![License](https://img.shields.io/github/license/gugarosa/opytimizer.svg)](https://github.com/gugarosa/opytimizer/blob/master/LICENSE)

## Welcome to Opytimizer.
Did you ever reach a bottleneck in your computational experiments? Are you tired of selecting suitable parameters for a chosen technique? If yes, Opytimizer is the real deal! This package provides an easy-to-go implementation of meta-heuristic optimizations. From agents to a search space, from internal functions to external communication, we will foster all research related to optimizing stuff.

Use Opytimizer if you need a library or wish to:
* Create your own optimization algorithm.
* Design or use pre-loaded optimization tasks.
* Mix-and-match different strategies to solve your problem.
* Because it is fun to optimize things.

Read the docs at [opytimizer.recogna.tech](opytimizer.recogna.tech).

Opytimizer is compatible with: **Python 3.6-3.7** and **PyPy 3.5**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy, if you wish to read the code and bump yourself into, just follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**, call us.

---

## Getting started: 60 seconds with Opytimizer

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, chose your subpackage and follow the example. We have high-level examples for most tasks we could think of.

Or if you wish to learn even more, please take a minute:

Opytimizer is based on the following structure, and you should pay attention to its tree:

```
- opytimizer
    - core
        - agent
        - function
        - optimizer
        - space
    - math
        - distribution
        - random
    - optimizers
        - fpa
        - pso
    - spaces
        - hyper
        - search
    - utils
        - history
        - logging
```

### Core

Core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basic of our structure. They should provide variables and methods that will help to construct other modules. It is composed by the following submodules:

```agent```: Submodule responsible for handling Agent's class. This can be referred as the particle, the individual or any single instance that corresponds to the basis of an optimization task.

```function```: When working with optimization, we need to evaluate our task into something. Function is the module ready-to-go to evaluate internal or external libraries functions.

```optimizer```: The outlier of optimization techniques. This serves as a foundation for creating more specific optimizers.

```space```: The space can be understood as the agents' house. In other words, is the entity responsible for holding agents, bounds and iterations to perform the optimization task.

### Math

Just because we are computing stuff, it does not means that we do not need math. Math is the mathematical package, containing low level math implementations. From random numbers to distributions generation, you can find your needs on this module.

```distribution```: Package used to handle distributions generation.

```random```: Package used to handle random numbers generation.

### Optimizers

This is why we are called Opytimizer. This is the heart of the heuristics, where you can find a broad number of meta-heuristics, optimization techniques, anything that can be called as an optimizer. Investigate over any module for more information.

```fpa```: Flower Pollination Algorithm.

```pso```: Particle Swarm Optimization.

### Spaces

One can see the space as the place that agents will update their positions and evaluate a fitness function. However, newest approaches may consider a different type of space. Thinking on that, we are glad to support diverse space implementations.

```hyper```: An hypercomplex space. It's the ideal space when dealing with complex, quaternions or octonions numbers.

```search```: A classic search space. The traditional way we have been optimizing things over the last years.

### Utils

This is an utilities package. Common things shared across the application should be implemented here. It is better to implement once and use as you wish than re-implementing the same thing over and over again.

```history```: A history class in order to save Opytimizer outputs. It servers as an interesting object for futher analysis of your experiment.

```logging```: Logging tools to track the progress of the optimization task.

---

## Installation

We belive that everything have to be easy. Not difficult or daunting, Opytimizer will be the one-to-go package that you will need, from the very first instalattion to the daily-tasks implementing needs. If you may, just run the following under your most preferende Python environment (raw, conda, virtualenv, whatever)!:

```Python
pip install .
```

---

## Environment configuration

Note that sometimes, there is a need for an additional implementation. If needed, from here you will be the one to know all of its details.

### Ubuntu

No specific additional commands needed.

### Windows

No specific additional commands needed.

### MacOS

No specific additional commands needed.

---

## Support

We know that we do our best, but it's inevitable to acknowlodge that we make mistakes. If you every need to report a bug, report a problem, talk to us, please do so! We will be avaliable at our bests at this repository or recogna@unesp.br.

---
