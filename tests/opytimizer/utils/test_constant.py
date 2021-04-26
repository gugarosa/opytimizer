import sys

from opytimizer.utils import constant


def test_constant_constants():
    assert constant.EPSILON == 1e-32

    assert constant.FLOAT_MAX == sys.float_info.max

    assert constant.HISTORY_KEYS == ['agents', 'best_agent', 'local_position']

    assert constant.LIGHT_SPEED == 3e5

    assert constant.FUNCTION_N_ARGS == {
        'SUM': 2,
        'SUB': 2,
        'MUL': 2,
        'DIV': 2,
        'EXP': 1,
        'SQRT': 1,
        'LOG': 1,
        'ABS': 1,
        'SIN': 1,
        'COS': 1
    }

    assert constant.TEST_EPSILON == 100

    assert constant.TOURNAMENT_SIZE == 2
