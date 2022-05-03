import opytimizer
from opytimizer.core import function
from opytimizer.optimizers.swarm import pso
from opytimizer.spaces import search
from opytimizer.utils import callback, history


def test_opytimizer_space():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    assert type(new_opytimizer.space).__name__ == "SearchSpace"


def test_opytimizer_space_setter():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    try:
        space.built = False
        new_opytimizer.space = space
    except:
        space.built = True
        new_opytimizer.space = space

    assert type(new_opytimizer.space).__name__ == "SearchSpace"


def test_opytimizer_optimizer():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    assert type(new_opytimizer.optimizer).__name__ == "PSO"


def test_opytimizer_optimizer_setter():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    try:
        optimizer.built = False
        new_opytimizer.optimizer = optimizer
    except:
        optimizer.built = True
        new_opytimizer.optimizer = optimizer

    assert type(new_opytimizer.optimizer).__name__ == "PSO"


def test_opytimizer_function():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    assert type(new_opytimizer.function).__name__ == "Function"


def test_opytimizer_function_setter():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    try:
        func.built = False
        new_opytimizer.function = func
    except:
        func.built = True
        new_opytimizer.function = func

    assert type(new_opytimizer.function).__name__ == "Function"


def test_opytimizer_history():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    assert type(new_opytimizer.history).__name__ == "History"


def test_opytimizer_history_setter():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()
    hist = history.History()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    try:
        new_opytimizer.history = 1
    except:
        new_opytimizer.history = hist

    assert type(new_opytimizer.history).__name__ == "History"


def test_opytimizer_iteration():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    assert new_opytimizer.iteration == 0


def test_opytimizer_iterations_setter():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    try:
        new_opytimizer.iteration = "a"
    except:
        new_opytimizer.iteration = 0

    assert new_opytimizer.iteration == 0

    try:
        new_opytimizer.iteration = -1
    except:
        new_opytimizer.iteration = 0

    assert new_opytimizer.iteration == 0


def test_opytimizer_total_iterations():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    assert new_opytimizer.total_iterations == 0


def test_opytimizer_total_iterations_setter():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    try:
        new_opytimizer.total_iterations = "a"
    except:
        new_opytimizer.total_iterations = 0

    assert new_opytimizer.total_iterations == 0

    try:
        new_opytimizer.total_iterations = -1
    except:
        new_opytimizer.total_iterations = 0

    assert new_opytimizer.total_iterations == 0


def test_opytimizer_evaluate_args():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    assert len(new_opytimizer.evaluate_args) == 2


def test_opytimizer_update_args():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    assert len(new_opytimizer.update_args) == 1


def test_opytimizer_evaluate():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()
    callbacks = callback.CallbackVessel([])

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    new_opytimizer.evaluate(callbacks)


def test_opytimizer_update():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()
    callbacks = callback.CallbackVessel([])

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    new_opytimizer.update(callbacks)


def test_opytimizer_start():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    new_opytimizer.start(n_iterations=1)


def test_opytimizer_save():
    space = search.SearchSpace(1, 1, 0, 1)
    func = function.Function(callable)
    optimizer = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(space, optimizer, func)

    new_opytimizer.save("out.pkl")


def test_opytimizer_load():
    new_opytimizer = opytimizer.Opytimizer.load("out.pkl")

    assert type(new_opytimizer).__name__ == "Opytimizer"
