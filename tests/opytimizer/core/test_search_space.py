import pytest

from opytimizer.core import search_space

s = search_space.SearchSpace(model_path='json/search_space_model.json') 
print(s.function.upper_bound)