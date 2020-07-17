import opytimizer.math.random as r

# Generating a binary random number array
b = r.generate_binary_random_number(size=10)
print(b)

# Generating a integer random number array
i = r.generate_integer_random_number(low=0, high=1, size=1)
print(i)

# Generating a random uniform number array
u = r.generate_uniform_random_number(low=0.0, high=1.0, size=1)
print(u)

# Generating a random gaussian number array
g = r.generate_gaussian_random_number(mean=0.5, variance=1.0, size=10)
print(g)
