import opytimizer.math.random as r

# Generates a binary random number array
b = r.generate_binary_random_number(size=10)
print(b)

# Generates an exponential random number array
e = r.generate_exponential_random_number(scale=1.0, size=10)
print(e)

# Generates an Erlang/gamma random number array
eg = r.generate_gamma_random_number(shape=1.0, scale=1.0, size=10)
print(eg)

# Generates an integer random number array
i = r.generate_integer_random_number(low=0, high=1, size=1)
print(i)

# Generates a random uniform number array
u = r.generate_uniform_random_number(low=0.0, high=1.0, size=1)
print(u)

# Generates a random gaussian number array
g = r.generate_gaussian_random_number(mean=0.5, variance=1.0, size=10)
print(g)
