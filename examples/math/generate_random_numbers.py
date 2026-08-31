import opytimizer.math.random as r

# Generates an integer array without the excluded value
i = r.integer(low=0, high=10, exclude=5, size=10)
print(i)
