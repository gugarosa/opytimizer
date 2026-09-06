# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

from opytimizer.optimizers.misc import gs


def test_gs():
    new_gs = gs.GS({"grid": [1, 2]})

    assert new_gs.grid == [1, 2]
