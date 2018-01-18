import pytest

from opytimizer.core import function
from opytimizer.utils import math

f = function.Function(expression='x0', lower_bound=1, upper_bound=2)
f.call()
f.evaluate()