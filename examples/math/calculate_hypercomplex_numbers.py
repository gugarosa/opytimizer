import numpy as np
import opytimizer.math.hypercomplex as h

a = np.ones((2, 1))
lb = np.array([-5, -5])
ub = np.array([-2, -2])
print(a)
print(h.norm(a))
print(h.span(a, lb, ub))