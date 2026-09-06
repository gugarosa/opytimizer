# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import opytimizer.math.general as g

individuals = [1, 2, 3, 4]

for pair in g.n_wise(individuals, 2):
    print(f"Pair: {pair}")

selected = g.tournament_selection(individuals, 2)

print(f"Selected: {selected}")
