import numpy as np

from opytimizer.math import hyper


def test_span_maps_zero_and_unit_norm_to_bounds():
    array = np.array([[0.0, 0.0], [1.0, 1.0]])

    spanned = hyper.span(array, [10], [20])

    np.testing.assert_allclose(spanned, [10, 20])


def test_span_to_hyper_value_transforms_objective_input():
    @hyper.span_to_hyper_value([10], [20])
    def objective(x):
        return np.sum(x)

    result = objective(np.array([[0.5], [0.5]]))

    assert result == 30
