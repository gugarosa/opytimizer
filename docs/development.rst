Extending and contributing
==========================

Keep the existing boundaries
----------------------------

``Space`` owns the population, an ``Optimizer`` strategy performs numerical
updates, ``Opytimizer`` controls the lifecycle, and callbacks observe or
deliberately modify that lifecycle. Use these existing extension points before
introducing a new interface, registry, configuration framework, or result type.

The base ``Optimizer`` is intentionally concrete. Its ``compile`` and ``update``
hooks do nothing, while ``evaluate`` supplies ordinary scalar minimization.
Subclasses only need to override the responsibilities they actually change.

A minimal optimizer
-------------------

.. testcode:: extension

   import numpy as np

   from opytimizer import Opytimizer
   from opytimizer.core import Optimizer
   from opytimizer.spaces import SearchSpace

   class HalvingOptimizer(Optimizer):
       """Move every component halfway toward zero.

       """

       def update(self, space):
           for agent in space.agents:
               agent.position *= 0.5

   space = SearchSpace(2, 2, [-1, -1], [1, 1])
   for agent in space.agents:
       agent.fill_with_static([0.8, -0.4])

   model = Opytimizer(
       space, HalvingOptimizer(), lambda position: float(np.sum(position**2))
   )
   model.start(2)
   np.testing.assert_allclose(model.space.best_agent.position, [[0.2], [-0.1]])
   np.testing.assert_allclose(model.space.best_agent.fit, 0.05)

This strategy does not need a custom constructor or auxiliary state. When
additional state is necessary, allocate it in ``compile(self, space)`` using the
actual population and position dimensions. Recompilation resets such state; the
driver does not recompile on every run.

Methods ``evaluate`` and ``update`` receive model attributes by their parameter
names. Common names are ``space``, ``function``, ``iteration``, ``n_iterations``,
and ``total_iterations``. Declare named positional parameters, not variadic or
keyword-only parameters. Defaults are not a substitute for model attributes.
Keep algorithm configuration on the optimizer rather than adding unrelated
parameters to these hooks.

Validate a new parameter's meaningful domain at configuration and use boundaries,
especially when public mutation remains supported. ``Optimizer.build`` accepts
mappings and applies attribute overrides; it does not reject unknown names or
provide a universal algorithm schema.

Reuse and state ownership
-------------------------

Use cooperative inheritance where the responsibility is genuinely shared.
For example, RPSO and VPSO reuse ``PSO.compile`` for common buffers and initialize
only their additional state. Do not duplicate that parent initialization or add
delegating constructors that have no work of their own.

Preserve ``(n_variables, n_dimensions)`` position shapes and document any narrower
space compatibility an algorithm requires. The shared representation alone does
not prove every algorithm supports every space.

Be explicit about current positions, incumbent fitness, personal bests, and global
bests. Do not silently change one convention to another in a cleanup. Similarly,
do not add generated equality/hash behavior to mutable NumPy-bearing containers
merely to convert them to dataclasses.

Objective and callback arguments are live objects. Avoid caching their values
across hooks, hiding mutation behind convenience wrappers, or silently substituting
success-shaped defaults after a failure.

Docstrings and types
--------------------

The canonical `code conventions <https://github.com/gugarosa/opytimizer/blob/main/CONVENTIONS.md>`_
adapt the cpmux/phitrain rules to this numerical library.
They specify headers, modern typing/import syntax, Google-style docstrings,
error prose, comments, and logical phase spacing.

Regular classes keep a single-sentence summary and document constructor arguments
on ``__init__``. Private helpers and concrete lifecycle overrides have no
docstrings. Keep shared contracts on their public base declarations.

Do not delete useful parameter meaning, array shapes, ownership details, or
scientific references to meet docstring placement rules. Preserve that information
in constructor notes, module documentation, or public guides.
Use actual model forward references rather than unrelated type variables.
An annotation is not runtime validation.

Modern Python syntax is useful when it removes redundant knowledge or clarifies a
contract. Keep all changes compatible with the declared Python support range;
do not turn a local cleanup into a language-version or tooling migration.

Executable examples and checks
------------------------------

The usage and extension examples are executed by Sphinx's built-in doctest
builder. Assert observable values, shapes, lifecycle effects, and invariants;
avoid brittle exact random trajectories or assertions about untouched initialized
state.

Use the repository's existing tools:

.. code-block:: text

   uv run pytest
   uv run pre-commit run --all-files
   uv run --group docs sphinx-build -b html -W --keep-going docs docs/_build/html
   uv run --group docs sphinx-build -b doctest -W --keep-going docs docs/_build/doctest

Start with the relevant tests and expand according to the affected surface.
Documentation builds and executable examples are CI gates alongside the runtime
test matrix. The pytest convention checks guard headers, imports, docstring
placement, diagnostic form, and mechanically checkable comment rules.
Semantic phase boundaries and meaningful prose still need human review.
No new runtime dependency is needed for this workflow.

Design references
-----------------

These are references for API discipline, not dependencies or APIs to copy wholesale.

* `SciPy 1.18.1 <https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_minimize.py#L197-L219>`_
  documents parameter-name-sensitive callback dispatch as a public contract.
* `scikit-learn 1.9.0 <https://github.com/scikit-learn/scikit-learn/blob/1.9.0/doc/developers/develop.rst#L122-L153>`_
  distinguishes configuration and learned state. Its estimator/cloning rules
  serve that ecosystem and are not requirements for this optimizer API.
* `Optuna 4.9.0 <https://github.com/optuna/optuna/blob/v4.9.0/optuna/study/study.py#L414-L451>`_
  provides state-owning, ``None``-returning orchestration with per-call callbacks.
* `pymoo 0.6.2 <https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/core/algorithm.py#L337-L353>`_
  uses an explicit algorithm lifecycle with optional no-op hooks. Its functional
  frontend's copying policy is different from Opytimizer's live-instance policy.
