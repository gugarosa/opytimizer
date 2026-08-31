from opytimizer.optimizers.misc import gs


def test_gs():
    new_gs = gs.GS({"grid": [1, 2]})

    assert new_gs.grid == [1, 2]
