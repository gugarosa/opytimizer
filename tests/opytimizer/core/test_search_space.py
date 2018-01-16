import pytest

from opytimizer.core import search_space
from opytimizer.utils import math

s = search_space.SearchSpace(model_path='json/search_space_model.json') 
s.function.instanciate()
s.agent[0].position[0] = [-1,-1]
s.agent[0].position[1] = [-1,-1]
s.evaluate()
print(s.agent[0].fit)