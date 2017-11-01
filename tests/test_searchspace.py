import opytimizer.core.search_space as SearchSpace

s = SearchSpace.SearchSpace(5, 10)
s.initialize('PSO', 'path.txt')
print(s.w)