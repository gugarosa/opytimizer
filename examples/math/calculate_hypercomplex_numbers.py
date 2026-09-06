# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np

import opytimizer.math.hyper as h

a = np.ones((2, 4))
print(f"Array: {a}")

lb = np.array([-5, -5])
ub = np.array([-2, -2])

span = h.span(a, lb, ub)
print(f"Spanned Array: {span}")
