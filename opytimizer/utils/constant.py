# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Define numerical safeguards and expression arities.

"""

import sys

# Avoid exact-zero denominators and logarithm arguments
EPSILON = 1e-32

# Rank unevaluated candidates behind ordinary finite objective values
FLOAT_MAX = sys.float_info.max

LIGHT_SPEED = 3e5

FUNCTION_N_ARGS = {
    "SUM": 2,
    "SUB": 2,
    "MUL": 2,
    "DIV": 2,
    "EXP": 1,
    "SQRT": 1,
    "LOG": 1,
    "ABS": 1,
    "SIN": 1,
    "COS": 1,
}

TEST_EPSILON = 100
