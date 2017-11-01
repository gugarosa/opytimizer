import numpy as np

"""
A random numbers generate module.
Differents distributions are used in order to generate numbers.
"""

def GenerateUniformRandomNumber(low, high, size=None):
    # Generates a random number based on an uniform distribution
    r = np.random.uniform(low, high, size)
    return r

def GenerateGaussianRandomNumber(mean, variance, size=None):
    # Generates a random number based on a gaussian distribution
    r = np.random.normal(mean, variance, size)
    return r
