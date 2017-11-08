import opytimizer.core.search_space as SearchSpace
import opytimizer.utils.random as Random

#s = SearchSpace.SearchSpace(5, 10)
#s.initialize('PSO', 'path.txt')

r = Random.GenerateGaussianRandomNumber(0, 0.1, 10)
print(r)