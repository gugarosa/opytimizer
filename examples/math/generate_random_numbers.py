# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import opytimizer.math.random as r

i = r.integer(low=0, high=10, exclude=5, size=10)
print(i)
