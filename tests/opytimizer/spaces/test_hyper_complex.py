from opytimizer.spaces import hyper_complex


def test_hyper_complex_initialize_agents():
    new_hyper_complex_space = hyper_complex.HyperComplexSpace(1, 1, 1)

    assert new_hyper_complex_space.agents[0].position[0][0] > 0


def test_hyper_complex_clip_by_bound():
    new_hyper_complex_space = hyper_complex.HyperComplexSpace(1, 1, 1)

    new_hyper_complex_space.agents[0].position[0][0] = 20

    new_hyper_complex_space.clip_by_bound()

    assert new_hyper_complex_space.agents[0].position[0][0] != 20
