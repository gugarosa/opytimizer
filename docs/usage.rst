Usage and lifecycle
===================

Opytimizer minimizes a scalar objective over a mutable population. The objective
receives a live position array with shape ``(n_variables, n_dimensions)``.
``SearchSpace`` uses one dimension per variable. Avoid mutating an objective's
input unless the application deliberately relies on that side effect.

An ordinary optimization
------------------------

.. testcode:: usage

   import numpy as np

   from opytimizer import Opytimizer
   from opytimizer.optimizers.swarm import PSO
   from opytimizer.spaces import SearchSpace

   def sphere(position):
       """Compute the squared Euclidean norm.

       Args:
           position: Candidate position array.

       Returns:
           Scalar fitness to minimize.

       """

       return float(np.sum(position**2))

   np.random.seed(7)
   space = SearchSpace(5, 2, [-1, -1], [1, 1])
   optimizer = PSO({"w": 0.7, "c1": 1.7, "c2": 1.7})
   model = Opytimizer(space, optimizer, sphere, save_agents=True)

   assert model.start(3) is None
   assert model.total_iterations == 3
   assert model.space.best_agent.position.shape == (2, 1)
   assert np.isfinite(model.space.best_agent.fit)

``start`` changes the supplied objects in place; it does not return a separate
result object. Use ``model.space.best_agent`` for the best recorded solution.
Some optimizers maintain different candidate-state conventions. For example,
PSO stores a particle's personal-best fitness in ``agent.fit`` and its matching
position in ``optimizer.local_position``; the current ``agent.position`` can be
worse.

The model does not clone its space or optimizer. Construct a new pair for an
independent task. Constructing another model with the same optimizer compiles
that shared optimizer again and may reset its buffers.

History shapes
--------------

.. testcode:: usage

   positions, fitness = model.history.get_convergence("best_agent")
   assert positions.shape == (2, 3)
   assert fitness.shape == (3,)
   assert np.all(fitness[1:] <= fitness[:-1])

   particle_positions, particle_fitness = model.history.get_convergence(
       "agents", index=0
   )
   assert particle_positions.shape == (2, 3)
   assert particle_fitness.shape == (3,)

History records are appended lazily. ``agents`` is not recorded unless
``save_agents=True``; requesting an absent key raises ``AttributeError`` rather
than returning an empty success-shaped result.

For ``agents`` and ``best_agent``, ``get_convergence`` returns a
``(positions, fitness)`` pair. With an integer agent index, position matrices are
concatenated horizontally: the shape is
``(n_variables, n_records * n_dimensions)``. Fitness has shape ``(n_records,)``.
Other keys return one array concatenated with ``numpy.hstack``. Raw records stay
available as history attributes.

Position snapshots are converted to independent lists. Arbitrary custom values
passed to ``History.dump`` are retained as supplied, so copy mutable custom data
when an independent snapshot is required.

Callbacks and repeated runs
---------------------------

.. testcode:: usage

   from opytimizer.utils.callback import Callback

   class Iterations(Callback):
       """Collect completed iteration counters.

       """

       def __init__(self):
           """Initialize the counter collection.

           """

           self.seen = []

       def on_iteration_end(self, iteration, opt_model):
           self.seen.append(iteration)

   recorder = Iterations()
   model.start(2, callbacks=[recorder])
   assert recorder.seen == [4, 5]
   assert model.total_iterations == 5
   assert model.iteration == 1
   assert len(model.history.best_agent) == 5
   assert len(model.history.time) == 2

Compilation happens once during ``Opytimizer`` construction. Each ``start``:

1. Validates its non-negative integer budget, then calls ``on_task_begin``.
2. Evaluates the initial population between evaluation-before/after hooks.
3. For each iteration, increments ``total_iterations``, sets the zero-based
   ``iteration``, and calls ``on_iteration_begin``.
4. Calls the update-before hook, optimizer update, and update-after hook, then
   clips positions to bounds.
5. Evaluates again, appends iteration history, and calls ``on_iteration_end``.
6. Calls ``on_task_end`` on normal completion, then appends elapsed run time.

Every hook runs in callback sequence order. Callbacks receive live objects.
``on_update_after`` runs before the driver's clipping, while
``on_iteration_end`` sees evaluated state and recorded iteration history.

Callbacks are supplied per invocation and are not automatically reused.
``start(0)`` still performs task hooks and initial evaluation, but no updates.
The loop restarts ``iteration`` at zero; before the first update it retains its
previous value. ``total_iterations`` is cumulative and is incremented before
iteration work, not after it.

Repeated calls continue optimizer state but can restart iteration-local schedules
and add objective evaluations. They are not generally equivalent to one longer
call, especially for adaptive algorithms or stochastic objectives.

Exceptions propagate without rollback. ``on_task_end`` is not called after an
objective or callback failure, and the current run's elapsed history is then
absent. Own external resources with context managers rather than relying on a
task-end callback for guaranteed cleanup.

Trusted checkpoints
-------------------

.. warning::

   Dill is pickle-based and can execute code while loading. Load only trusted
   checkpoints. Keep the software environment compatible; checkpoints are not a
   version-independent interchange format.

.. testcode:: usage

   from pathlib import Path
   from tempfile import TemporaryDirectory

   from opytimizer.utils.callback import CheckpointCallback

   with TemporaryDirectory() as directory:
       path = Path(directory) / "model.pkl"
       model.start(1, callbacks=[CheckpointCallback(path, frequency=1)])
       checkpoint = path.with_name("iter_6_model.pkl")
       assert checkpoint.is_file()

       restored = Opytimizer.load(checkpoint)
       assert restored.total_iterations == 6
       best_before_resume = restored.space.best_agent.fit
       restored.start(1)
       assert restored.total_iterations == 7
       assert restored.space.best_agent.fit <= best_before_resume
       assert not path.with_name("iter_7_model.pkl").exists()

The driver retains space, optimizer, objective, history, and counters. It does
not register the callback sequence as model state; pass callbacks again when
resuming. A checkpoint taken at iteration end precedes that run's task-end hook
and elapsed-time record. ``CheckpointCallback(frequency=0)`` is disabled.

For an explicit checkpoint outside a callback, use ``model.save(path)`` and
``Opytimizer.load(path)`` with a string or text path-like object. Parent
directories must already exist.

Reproducibility
---------------

Seed NumPy before creating spaces or optimizer state to control the existing
module-level random stream. Stochastic objectives need their own randomness
policy as well. Shared model instances and a process-global random stream are
not an isolation mechanism for concurrent tasks. The driver does not capture or
restore that process-global RNG state in its checkpoints.

A seed is not a promise of identical trajectories across algorithm corrections,
dependency versions, or changes in objective-call order.
