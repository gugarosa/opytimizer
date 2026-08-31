import opytimizer.math.distribution as d

# Generates a Lévy distribution
l = d.generate_levy_distribution(beta=0.5, size=10)
print(l)
