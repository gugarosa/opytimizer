# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from opytimizer import Opytimizer
from opytimizer.core import Optimizer, Space
from opytimizer.spaces import SearchSpace
from opytimizer.utils.callback import Callback


def sphere(x):
    return float(np.sum(x**2))


class StepOptimizer(Optimizer):
    def __init__(self):
        self.compiled = False

    def compile(self, space):
        self.compiled = True

    def update(self, space):
        for agent in space.agents:
            agent.position += 2


class Recorder(Callback):
    def __init__(self):
        self.events = []

    def on_task_begin(self, opt_model):
        self.events.append("task_begin")

    def on_task_end(self, opt_model):
        self.events.append("task_end")

    def on_iteration_begin(self, iteration, opt_model):
        self.events.append(f"iteration_begin:{iteration}")

    def on_iteration_end(self, iteration, opt_model):
        self.events.append(f"iteration_end:{iteration}")

    def on_evaluate_before(self, *evaluate_args):
        self.events.append("evaluate_before")

    def on_evaluate_after(self, *evaluate_args):
        self.events.append("evaluate_after")

    def on_update_before(self, *update_args):
        self.events.append("update_before")

    def on_update_after(self, *update_args):
        self.events.append("update_after")


def make_model(save_agents=False):
    space = SearchSpace(1, 1, 0, 1)
    space.agents[0].position[:] = 0.5
    optimizer = StepOptimizer()
    return Opytimizer(space, optimizer, sphere, save_agents), optimizer


def test_opytimizer_accepts_raw_callable_and_compiles_optimizer():
    model, optimizer = make_model()

    assert model.function is sphere
    assert optimizer.compiled
    assert len(model.evaluate_args) == 2
    assert len(model.update_args) == 1
    assert not hasattr(model.space, "built")
    assert not hasattr(model.optimizer, "built")


def test_opytimizer_validates_constructor_inputs():
    space = SearchSpace(1, 1, 0, 1)
    optimizer = StepOptimizer()

    with pytest.raises(TypeError):
        Opytimizer(object(), optimizer, sphere)
    with pytest.raises(TypeError):
        Opytimizer(space, object(), sphere)
    with pytest.raises(TypeError):
        Opytimizer(space, optimizer, 1)


def test_opytimizer_rejects_uninitialized_space():
    with pytest.raises(RuntimeError):
        Opytimizer(Space(), StepOptimizer(), sphere)


def test_opytimizer_validates_history_options_before_compiling():
    optimizer = StepOptimizer()

    with pytest.raises(TypeError, match="save_agents"):
        Opytimizer(SearchSpace(1, 1, 0, 1), optimizer, sphere, save_agents="yes")

    assert not optimizer.compiled


@pytest.mark.parametrize(
    "budget,error",
    [
        (-1, ValueError),
        (np.int64(-1), ValueError),
        (1.5, TypeError),
        (np.float64(2), TypeError),
        ("2", TypeError),
        (None, TypeError),
    ],
)
@pytest.mark.parametrize("resume", [False, True])
def test_opytimizer_rejects_invalid_budgets_before_side_effects(budget, error, resume):
    model, _ = make_model()
    if resume:
        model.start()
    calls = []
    recorder = Recorder()
    before = (
        model.iteration,
        model.total_iterations,
        getattr(model, "n_iterations", None),
    )
    history = deepcopy(vars(model.history))
    model.function = lambda x: calls.append(x.copy()) or sphere(x)

    with pytest.raises(error, match="n_iterations"):
        model.start(budget, [recorder])

    assert calls == []
    assert recorder.events == []
    assert before == (
        model.iteration,
        model.total_iterations,
        getattr(model, "n_iterations", None),
    )
    assert vars(model.history) == history


@pytest.mark.parametrize("budget", [0, 1, np.int64(2)])
def test_opytimizer_preserves_valid_integer_budgets(budget):
    model, _ = make_model()

    model.start(budget)

    assert model.n_iterations is budget
    assert model.total_iterations == int(budget)


def test_opytimizer_reuses_state_and_scopes_callbacks_to_each_run(monkeypatch):
    model, optimizer = make_model()
    recorder = Recorder()

    def unexpected_compile(space):
        pytest.fail("start must not recompile optimizer state")

    monkeypatch.setattr(optimizer, "compile", unexpected_compile)
    model.start(2, [recorder])
    events = recorder.events.copy()
    model.start(3)

    assert recorder.events == events
    assert model.iteration == 2
    assert model.total_iterations == 5
    assert len(model.history.best_agent) == 5
    assert len(model.history.time) == 2


def test_opytimizer_measures_elapsed_time_with_a_monotonic_clock(monkeypatch):
    model, _ = make_model()
    ticks = iter([10.0, 12.5])

    with monkeypatch.context() as context:
        context.setattr("opytimizer.opytimizer.time.perf_counter", lambda: next(ticks))
        model.start(0)

    assert model.history.time == [2.5]


def test_opytimizer_propagates_objective_failures_without_task_end():
    model, _ = make_model()
    recorder = Recorder()

    def fail(position):
        raise RuntimeError("`objective` failed.")

    model.function = fail

    with pytest.raises(RuntimeError, match=r"`objective` failed\."):
        model.start(1, [recorder])

    assert recorder.events == ["task_begin", "evaluate_before"]
    assert not hasattr(model.history, "time")


def test_opytimizer_does_not_discard_falsey_callback_sequences():
    class FalseyCallbacks(list):
        def __bool__(self):
            return False

    model, _ = make_model()
    recorder = Recorder()

    model.start(0, FalseyCallbacks([recorder]))

    assert recorder.events == [
        "task_begin",
        "evaluate_before",
        "evaluate_after",
        "task_end",
    ]


def test_opytimizer_preserves_callback_order_clipping_and_history():
    model, _ = make_model(save_agents=True)
    recorder = Recorder()

    model.start(1, [recorder])

    assert recorder.events == [
        "task_begin",
        "evaluate_before",
        "evaluate_after",
        "iteration_begin:1",
        "update_before",
        "update_after",
        "evaluate_before",
        "evaluate_after",
        "iteration_end:1",
        "task_end",
    ]
    assert np.array_equal(model.space.agents[0].position, [[1]])
    assert model.iteration == 0
    assert model.total_iterations == 1
    assert model.n_iterations == 1
    assert len(model.history.agents) == 1
    assert len(model.history.best_agent) == 1
    assert len(model.history.time) == 1


def test_opytimizer_dispatches_callbacks_in_list_order():
    calls = []

    class OrderedCallback(Callback):
        def __init__(self, name):
            self.name = name

        def on_task_begin(self, opt_model):
            calls.append(("task_begin", self.name))

        def on_evaluate_before(self, *evaluate_args):
            calls.append(("evaluate_before", self.name))

        def on_evaluate_after(self, *evaluate_args):
            calls.append(("evaluate_after", self.name))

        def on_task_end(self, opt_model):
            calls.append(("task_end", self.name))

    model, _ = make_model()
    model.start(0, [OrderedCallback("first"), OrderedCallback("second")])

    assert calls == [
        ("task_begin", "first"),
        ("task_begin", "second"),
        ("evaluate_before", "first"),
        ("evaluate_before", "second"),
        ("evaluate_after", "first"),
        ("evaluate_after", "second"),
        ("task_end", "first"),
        ("task_end", "second"),
    ]


def test_opytimizer_evaluate_and_update_without_callbacks():
    model, _ = make_model()

    model.evaluate()
    model.update()

    assert model.space.best_agent.fit == 0.25
    assert np.array_equal(model.space.agents[0].position, [[1]])


def test_opytimizer_save_and_load():
    model, _ = make_model()
    path = Path(f"opytimizer-{os.getpid()}.pkl")

    try:
        model.save(str(path))
        loaded = Opytimizer.load(str(path))
        assert isinstance(loaded, Opytimizer)
        assert loaded.function(np.array([2])) == 4
    finally:
        path.unlink(missing_ok=True)
