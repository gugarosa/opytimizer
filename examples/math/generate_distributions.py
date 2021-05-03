import opytimizer.math.distribution as d

# Generates a Bernoulli distribution
b = d.generate_bernoulli_distribution(prob=0.5, size=10)
print(b)

# Generates a choice distribution
c = d.generate_choice_distribution(n=10, probs=None, size=10)
print(c)

# Generates a Lévy distribution
l = d.generate_levy_distribution(beta=0.5, size=10)
print(l)
