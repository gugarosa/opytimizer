from opytimizer.optimizers.misc import gs


def test_gs():
    new_gs = gs.GS()

    assert new_gs.built is True
