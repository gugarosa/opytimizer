from opytimizer.functions.multi_objective import standard


def test_standard_functions():
    new_standard = standard.MultiObjectiveFunction([])

    assert type(new_standard.functions) == list


def test_standard_functions_setter():
    new_standard = standard.MultiObjectiveFunction([])

    try:
        new_standard.functions = None
    except:
        new_standard.functions = [1, 2]

    assert len(new_standard.functions) == 2


def test_standard_call():
    def square(x):
        return x**2

    assert square(2) == 4

    def cube(x):
        return x**3

    assert cube(2) == 8

    new_standard = standard.MultiObjectiveFunction(functions=[square, cube])

    assert new_standard(2) == [4, 8]
