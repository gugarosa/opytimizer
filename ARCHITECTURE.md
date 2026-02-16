# Opytimizer — Architecture Guide

> **Version:** 3.1.4 · **License:** Apache 2.0 · **Python:** 3.6+
> A nature-inspired meta-heuristic optimization framework.

---

## 1. Overview

Opytimizer is a modular Python framework for nature-inspired (meta-heuristic) optimization. It provides a plug-and-play architecture where **Agents** explore a **Search Space** guided by an **Optimizer** to minimize an **Objective Function**. The library ships **85+ optimization algorithms** spanning seven families, from classic Particle Swarm and Genetic Algorithms to physics-based Simulated Annealing and socially-inspired Brain Storm Optimization.

The design philosophy is focused on **minimization** — all fitness comparisons use `<` (less-is-better). Users who need maximization should negate their objective function.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Opytimizer (entry point)                  │
│  Orchestrates the full optimization loop:                        │
│  evaluate → update → clip → record history → repeat             │
├────────────┬──────────────┬─────────────┬────────────────────────┤
│   Space    │  Optimizer   │  Function   │  Utilities             │
│  (agents + │  (algorithm  │  (objective │  (callbacks, history,  │
│   bounds)  │   logic)     │   wrapper)  │   logging, math, viz)  │
└────────────┴──────────────┴─────────────┴────────────────────────┘
```

### Minimal Usage Pattern

```python
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

space     = SearchSpace(n_agents=20, n_variables=2, lower_bound=[-10,-10], upper_bound=[10,10])
optimizer = PSO()
function  = Function(lambda x: (x**2).sum())

opt = Opytimizer(space, optimizer, function)
opt.start(n_iterations=1000)
```

---

## 3. Core Module (`opytimizer/core/`)

The core module defines the seven foundational abstractions upon which everything else is built.

### 3.1 Agent (`agent.py`)

The atomic unit of any optimization. Each agent represents a **candidate solution**.

| Property | Description |
|---|---|
| `position` | `np.ndarray` of shape `(n_variables, n_dimensions)` — the decision variable values |
| `fit` | Scalar fitness value (initialized to `sys.float_info.max`) |
| `lb` / `ub` | Per-variable lower and upper bounds |
| `mapping` | Human-readable names for variables (e.g. `["learning_rate", "n_hidden"]`) |
| `ts` | Unix timestamp of when the agent was last updated as best |

Key methods:
- **`fill_with_uniform()`** — initialize positions uniformly within bounds
- **`fill_with_binary()`** — initialize with binary {0,1} values
- **`fill_with_static(values)`** — set positions to specific values (ignores bounds)
- **`clip_by_bound()`** — clamp all variables to `[lb, ub]`

### 3.2 Space (`space.py`)

The base class for all search spaces. It manages a **population of agents** and tracks the **global best agent**.

| Property | Description |
|---|---|
| `n_agents` | Population size |
| `n_variables` | Number of decision variables per agent |
| `n_dimensions` | Dimensionality of each variable (1 for standard, >1 for hypercomplex) |
| `agents` | `List[Agent]` — the population |
| `best_agent` | `Agent` — global best solution found so far |
| `built` | Boolean flag for readiness |

The `build()` method creates agents (`_create_agents()`) and initializes them (`_initialize_agents()`, which subclasses override). After building, the space is ready for the optimizer.

### 3.3 Function (`function.py`)

A thin wrapper around a user-provided callable. It validates the callable has exactly **one argument** (the position array), stores it, and makes the Function object itself callable (`__call__` delegates to `pointer`).

### 3.4 Optimizer (`optimizer.py`)

The base class for all 85+ optimization algorithms. It defines the three-phase contract:

1. **`build(params)`** — sets hyperparameters dynamically via `setattr`
2. **`compile(space)`** — pre-allocates algorithm-specific state (velocities, memories, etc.)
3. **`evaluate(space, function)`** — evaluates all agents and updates `best_agent`
4. **`update(...)`** — applies the algorithm's position-update rule

The `evaluate` base implementation is a simple greedy update: for each agent, compute fitness and replace `best_agent` if improved. Algorithms that need custom evaluation (e.g. PSO's local-best tracking, GP's tree evaluation) override this method.

### 3.5 Node (`node.py`)

A binary-tree node used by **Genetic Programming (GP)** and its variants. Each node is either:
- **TERMINAL** — holds a value (`np.ndarray`) representing a variable
- **FUNCTION** — holds a name (`SUM`, `MUL`, `EXP`, `SIN`, etc.) and two children

The `position` property recursively evaluates the tree, and `pre_order` / `post_order` properties provide traversal iterators. Supported function nodes: `SUM`, `SUB`, `MUL`, `DIV`, `EXP`, `SQRT`, `LOG`, `ABS`, `SIN`, `COS`.

### 3.6 Block (`block.py`)

Foundation for **graph-based (DAG) optimization**. Three types:
- `InputBlock` — entry point (identity function)
- `InnerBlock` — holds an arbitrary callable
- `OutputBlock` — exit point (identity function)

Blocks specify their `n_input` and `n_output` counts, which are validated when connecting edges.

### 3.7 Cell (`cell.py`)

A Directed Acyclic Graph (DAG) that composes Blocks into a computation graph using NetworkX's `DiGraph`. The `__call__` method performs a forward pass through all simple paths from input to output, collecting outputs from each path.

---

## 4. Spaces Module (`opytimizer/spaces/`)

Seven specialized search spaces extend the base `Space`:

| Space | Module | Description | Key Characteristics |
|---|---|---|---|
| **SearchSpace** | `search.py` | Standard continuous optimization | `n_dimensions=1`, uniform init |
| **BooleanSpace** | `boolean.py` | Binary / discrete optimization | Bounds fixed to [0,1], binary init |
| **GridSpace** | `grid.py` | Exhaustive grid search | Creates all grid-point combinations from `step` sizes; `n_agents` = grid size |
| **HyperComplexSpace** | `hyper_complex.py` | Quaternion / hypercomplex search | Multi-dimensional per variable (e.g., 4D quaternions) |
| **TreeSpace** | `tree.py` | Genetic Programming | Manages a list of `Node` trees alongside agents; GROW algorithm for tree init |
| **ParetoSpace** | `pareto.py` | Multi-objective (pre-loaded points) | Loads from pre-defined `data_points`; no bound clipping |
| **GraphSpace** | `graph.py` | DAG-based search (experimental) | Placeholder for graph-structured optimization |

---

## 5. Functions Module (`opytimizer/functions/`)

### 5.1 Function (core)
Single-objective wrapper — validates callable signature and provides `__call__`.

### 5.2 ConstrainedFunction (`constrained.py`)
Extends `Function` with a list of **constraint callables** and a **penalty factor**. When a constraint returns `False`, the fitness is penalized: `fitness += penalty * fitness`.

### 5.3 MultiObjectiveFunction (`multi_objective/standard.py`)
Wraps a list of objective functions. Returns a **list of fitness values** (one per objective) — suitable for Pareto-based methods like NDS.

### 5.4 MultiObjectiveWeightedFunction (`multi_objective/weighted.py`)
Extends `MultiObjectiveFunction` with a **weighted-sum scalarization strategy**, returning a single scalar: `z = Σ(wᵢ · fᵢ(x))`.

---

## 6. Optimizers Module (`opytimizer/optimizers/`)

The heart of the library — **85+ algorithms** organized into 7 families.

### 6.1 Swarm-Based (`swarm/`) — 29 algorithms

Inspired by collective behavior of animals, insects, and flocks.

| Algorithm | Class | Inspired By |
|---|---|---|
| ABC | Artificial Bee Colony | Honeybee foraging |
| ABO | African Buffalo Optimization | Buffalo herd movement |
| AF | Artificial Flora | Plant propagation |
| BA | Bat Algorithm | Bat echolocation |
| BOA | Butterfly Optimization | Butterfly foraging |
| BWO | Black Widow Optimization | Black widow spider mating |
| CS | Cuckoo Search | Cuckoo brood parasitism |
| CSA | Crow Search Algorithm | Crow food-caching |
| EHO | Elephant Herding Optimization | Elephant herding |
| FA | Firefly Algorithm | Firefly bioluminescence |
| FFOA | Fruit Fly Optimization | Fruit fly olfaction |
| FPA | Flower Pollination | Flower pollination |
| FSO | Fish School Optimization | Fish schooling |
| GOA | Grasshopper Optimization | Grasshopper swarming |
| JS / NBJS | Jellyfish Search | Jellyfish movement |
| KH | Krill Herd | Krill herding |
| MFO | Moth-Flame Optimization | Moth navigation |
| MRFO | Manta Ray Foraging | Manta ray feeding |
| PIO | Pigeon-Inspired Optimization | Pigeon navigation |
| PSO | Particle Swarm Optimization | Bird flocking / fish schooling |
| — AIWPSO | Adaptive Inertia Weight PSO | Adaptive inertia |
| — RPSO | Relativistic PSO | Relativistic velocity |
| — SAVPSO | Self-Adaptive Velocity PSO | Constrained velocity |
| — VPSO | Vertical PSO | Vertical velocity component |
| SBO | Satin Bowerbird Optimization | Bowerbird courtship |
| SCA | Sine Cosine Algorithm | Sine/cosine waves |
| SFO | Sailfish Optimization | Sailfish hunting |
| SOS | Symbiotic Organisms Search | Biological symbiosis |
| SSA | Salp Swarm Algorithm | Salp chain |
| SSO | Social Spider Optimization | Spider colony |
| STOA | Sooty Tern Optimization | Sooty tern migration |
| WAOA | Weighted Atom Orbits Algorithm | Atomic orbits |
| WOA | Whale Optimization | Whale bubble-net feeding |

### 6.2 Evolutionary (`evolutionary/`) — 11 algorithms (16 variants)

Based on biological evolution: selection, crossover, mutation.

| Algorithm | Class | Technique |
|---|---|---|
| BSA | Backtracking Search | Historical population memory |
| DE | Differential Evolution | Difference vector mutation |
| EP | Evolutionary Programming | Self-adaptive mutation |
| ES | Evolution Strategy | (μ, λ) / (μ + λ) strategies |
| FOA | Forest Optimization | Tree seeding, local/global seeding |
| GA | Genetic Algorithm | Roulette selection, BLX crossover, gaussian mutation |
| GP | Genetic Programming | Tree-based representation (GROW algorithm) |
| GSGP | Geometric Semantic GP | Semantic-aware operators |
| HS | Harmony Search | Musical improvisation (+ 5 variants: IHS, GHS, SGHS, NGHS, GOGHS) |
| IWO | Invasive Weed Optimization | Weed colonization |
| RRA | Raven Roosting Algorithm | Raven roosting behavior |

### 6.3 Science-Based (`science/`) — 20 algorithms

Inspired by physical, chemical, and astronomical phenomena.

| Algorithm | Class | Inspired By |
|---|---|---|
| AIG | Artificial Immune Gradient | Immune system + gradient |
| ASO | Atom Search Optimization | Atomic interactions |
| BH | Black Hole | Black hole absorption |
| CDO | Chernobyl Disaster Optimization | Nuclear radiation |
| EFO | Electromagnetic Field Optimization | EM fields |
| EO | Equilibrium Optimizer | Mass balance / equilibrium |
| ESA | Electro-Search Algorithm | Electrostatic attraction |
| GSA | Gravitational Search Algorithm | Newtonian gravity |
| HGSO | Henry Gas Solubility Optimization | Henry's gas law |
| LSA | Lightning Search Algorithm | Lightning propagation |
| MOA | Multi-verse Optimizer Algorithm | Multiverse cosmology |
| MVO | Multi-Verse Optimizer | White/black/worm holes |
| SA | Simulated Annealing | Metallurgic annealing |
| SMA | Slime Mould Algorithm | Slime mould oscillation |
| TEO | Thermal Exchange Optimization | Heat transfer |
| TWO | Tug of War Optimization | Tug of war mechanics |
| WCA | Water Cycle Algorithm | Water cycle (evaporation, rain) |
| WDO | Wind Driven Optimization | Wind dynamics |
| WEO | Water Evaporation Optimization | Water evaporation |
| WWO | Water Wave Optimization | Shallow water waves |

### 6.4 Population-Based (`population/`) — 12 algorithms

Inspired by animal group dynamics, predator-prey interactions, and ecosystem behavior.

| Algorithm | Class | Inspired By |
|---|---|---|
| AEO | Artificial Ecosystem Optimization | Ecosystem interactions |
| AO | Aquila Optimizer | Aquila eagle hunting |
| COA | Coyote Optimization | Coyote pack behavior |
| EPO | Emperor Penguin Optimizer | Penguin huddling |
| GCO | Germinal Center Optimization | Immune germinal centers |
| GWO | Grey Wolf Optimizer | Grey wolf social hierarchy |
| HHO | Harris Hawks Optimization | Harris hawk cooperative hunting |
| LOA | Lion Optimization | Lion pride dynamics |
| OSA | Owl Search Algorithm | Owl hunting behavior |
| PPA | Polar Bear Optimization (?) | Predator-prey interaction |
| PVS | Parasitism-Predation-Virus Search | Ecological interactions |
| RFO | Red Fox Optimization | Fox hunting strategies |

### 6.5 Social-Based (`social/`) — 6 algorithms

Inspired by human social behaviors and group dynamics.

| Algorithm | Class | Inspired By |
|---|---|---|
| BSO | Brain Storm Optimization | Brainstorming process |
| CI | Cohort Intelligence | Cohort learning behavior |
| ISA | Interior Search Algorithm | Interior design process |
| MVPA | Most Valuable Player Algorithm | Sports team dynamics |
| QSA | Queuing Search Algorithm | Queuing theory |
| SSD | Social Ski-Driver | Ski-driver social dynamics |

### 6.6 Boolean (`boolean/`) — 3 algorithms

Specialized for binary / discrete optimization problems.

| Algorithm | Class | Based On |
|---|---|---|
| BMRFO | Binary Manta Ray Foraging | MRFO with transfer functions |
| BPSO | Binary PSO | PSO with sigmoid transfer |
| UMDA | Univariate Marginal Distribution | Estimation of distribution |

### 6.7 Miscellaneous (`misc/`) — 6 algorithms

Classical and general-purpose methods.

| Algorithm | Class | Technique |
|---|---|---|
| AOA | Arithmetic Optimization | Arithmetic operators |
| CEM | Cross-Entropy Method | Cross-entropy sampling |
| DOA | Dice Optimization | Dice rolling mechanics |
| GS | Grid Search | Exhaustive grid evaluation |
| HC | Hill Climbing | Greedy local search |
| NDS | Non-Dominated Sorting | Pareto frontier ranking |

---

## 7. Math Module (`opytimizer/math/`)

Low-level mathematical utilities used across all optimizers.

### 7.1 `random.py` — Random number generators
- `generate_binary_random_number(size)` — uniform {0,1}
- `generate_uniform_random_number(low, high, size)` — uniform [low, high)
- `generate_gaussian_random_number(mean, variance, size)` — normal distribution
- `generate_integer_random_number(low, high, exclude_value, size)` — integer with optional exclusion
- `generate_exponential_random_number(scale, size)` — exponential distribution
- `generate_gamma_random_number(shape, scale, size)` — gamma (Erlang) distribution

### 7.2 `distribution.py` — Distribution generators
- `generate_bernoulli_distribution(prob, size)` — Bernoulli trials
- `generate_choice_distribution(n, probs, size)` — weighted random choice without replacement
- `generate_levy_distribution(beta, size)` — Lévy flight (used by Cuckoo Search, etc.)

### 7.3 `general.py` — General math functions
- `euclidean_distance(x, y)` — L2 norm between points
- `kmeans(x, n_clusters, max_iterations, tol)` — K-means clustering (used by BSO, etc.)
- `n_wise(x, size)` — iterate over pairs/n-tuples
- `tournament_selection(fitness, n, size)` — tournament selection (used by GP, GA)
- `weighted_wheel_selection(weights)` — roulette-wheel selection

### 7.4 `hyper.py` — Hypercomplex math
- `norm(array)` — norm across dimension axis
- `span(array, lb, ub)` — map hypercomplex values to real bounds
- `span_to_hyper_value(lb, ub)` — decorator that wraps an objective function to accept hypercomplex inputs

---

## 8. Utils Module (`opytimizer/utils/`)

### 8.1 `constant.py` — Global constants
| Constant | Value | Purpose |
|---|---|---|
| `EPSILON` | `1e-32` | Prevents division by zero |
| `FLOAT_MAX` | `sys.float_info.max` | Initial fitness for agents |
| `LIGHT_SPEED` | `3e5` | Used by Relativistic PSO |
| `FUNCTION_N_ARGS` | `dict` | Number of arguments per GP function node |
| `TEST_EPSILON` | `100` | Test pass threshold |

### 8.2 `exception.py` — Custom exceptions
Five exception types, all inheriting from a base `Error` class that logs to the logger:
- `ArgumentError` — wrong number of arguments
- `BuildError` — class not built before use
- `SizeError` — mismatched array sizes
- `TypeError` — wrong variable type
- `ValueError` — out-of-range values

### 8.3 `history.py` — Optimization history
The `History` class records per-iteration data via `dump(**kwargs)`. It dynamically creates list-based attributes for any key passed. Built-in parsing rules for:
- `agents` — stores `(position, fit)` tuples per agent
- `best_agent` — stores `(position, fit)` of the best
- `local_position` — stores local positions (e.g., PSO personal bests)

The `get_convergence(key, index)` method extracts time-series for plotting.

### 8.4 `callback.py` — Callback system
An event-driven hook system with 8 lifecycle events:

```
on_task_begin → [on_iteration_begin → on_update_before → UPDATE → on_update_after →
                 on_evaluate_before → EVALUATE → on_evaluate_after → on_iteration_end] × N
→ on_task_end
```

**Built-in callbacks:**
- `CheckpointCallback(file_path, frequency)` — periodic model serialization to disk
- `DiscreteSearchCallback(allowed_values)` — maps continuous positions to the nearest allowed discrete values before evaluation

The `CallbackVessel` class aggregates multiple callbacks and dispatches events to all of them.

### 8.5 `logging.py` — Logging infrastructure
Custom `Logger` class with a `to_file()` method that temporarily suppresses console output (logs only to the rotating file handler `opytimizer.log`). This is used during iteration loops to avoid console spam while still recording iteration details.

---

## 9. Visualization Module (`opytimizer/visualization/`)

### 9.1 `convergence.py`
Plots 2D convergence curves (`iteration × fitness`) for one or more variables. Accepts `*args` of arrays and optional labels.

### 9.2 `surface.py`
Renders 3D surface plots of benchmark functions using matplotlib's `plot_surface` + `plot_wireframe`. Expects points with shape `(3, n, n)`.

---

## 10. The Optimization Loop (`opytimizer.py`)

The `Opytimizer` class is the top-level orchestrator. Its `start(n_iterations, callbacks)` method runs:

```
1. compile(space)           # Optimizer pre-allocates state
2. evaluate(space, function) # Initial evaluation
3. FOR t = 1..n_iterations:
   a. update(...)           # Algorithm-specific position update
   b. clip_by_bound()       # Enforce bounds
   c. evaluate(...)         # Re-evaluate all agents
   d. history.dump(...)     # Record iteration data
4. history.dump(time=...)    # Record total time
```

**Introspection magic:** The `evaluate_args` and `update_args` properties use `inspect.signature` to dynamically resolve what arguments each optimizer's `evaluate()` and `update()` methods need (e.g., `space`, `function`, `iteration`), then pull the corresponding attributes from the `Opytimizer` instance. This means optimizers can declare whatever arguments they need and the orchestrator adapts automatically.

**Serialization:** `save(path)` / `load(path)` use `dill` (enhanced pickle) to serialize the entire optimization state, enabling checkpoint/resume workflows.

---

## 11. Design Patterns & Conventions

### 11.1 Property Validation
Every class uses Python properties with setters that perform **strict type and value validation**, raising custom exceptions. This defensive style catches configuration errors early rather than producing cryptic NumPy errors during optimization.

### 11.2 Build Pattern
Most objects follow a two-phase construction:
1. `__init__()` — set parameters, create empty structures
2. `build()` — allocate arrays, create agents, mark as `built=True`

The `Opytimizer` entry point validates that all three components (space, optimizer, function) are `built` before proceeding.

### 11.3 Template Method Pattern
The `Optimizer` base class defines the skeleton (`evaluate` → `update` → `clip`), while concrete optimizers override `update()`, `evaluate()`, and `compile()` as needed.

### 11.4 Dynamic Parameter Injection
`Optimizer.build(params)` iterates over a `params` dict and calls `setattr(self, k, v)` for each entry. This means users can override any algorithm hyperparameter at construction time:
```python
PSO(params={"w": 0.5, "c1": 2.0, "c2": 2.0})
```

---

## 12. Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations, linear algebra |
| `networkx` | DAG structure for Cell/GraphSpace |
| `matplotlib` | Convergence and surface plotting |
| `tqdm` | Progress bars for the optimization loop |
| `dill` | Model serialization (superset of pickle) |
| `opytimark` | Benchmark optimization functions |

---

## 13. Testing & Quality

- Tests live in `tests/opytimizer/` mirroring the package structure
- Uses `pytest` as the test runner with a `TEST_EPSILON = 100` fitness threshold
- Pre-commit hooks configured with `black` (line-length 88), `pylint`, and `pre-commit`
- CI via Travis CI (`.travis.yml`)

---

## 14. Project Lineage

Originally authored by **Gustavo H. de Rosa** at UNESP. Now maintained by the **Recogna Laboratory** at `github.com/recogna-lab/opytimizer`. The project is citable via arXiv (`1912.13002`) and Zenodo (`10.5281/zenodo.4594294`).

---

## 15. Integration Ecosystem

The `examples/integrations/` directory shows how to use Opytimizer as a hyperparameter tuner for:
- **PyTorch** — neural network hyperparameters
- **Scikit-Learn** — model selection and tuning
- **TensorFlow** — deep learning hyperparameters
- **Learnergy** — energy-based models
- **NALP** — natural language processing
- **OPFython** — Optimum-Path Forest classifiers
