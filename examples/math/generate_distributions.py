# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import opytimizer.math.distribution as d

samples = d.generate_levy_distribution(beta=0.5, size=10)
print(samples)
