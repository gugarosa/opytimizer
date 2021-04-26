from opytimizer.visualization import convergence


def test_convergence_plot():
    agent_pos = [[0.5, 0.4, 0.3], [0.5, 0.4, 0.3]]

    try:
        convergence.plot(agent_pos[0], agent_pos[1], labels=1)
    except:
        convergence.plot(agent_pos[0], agent_pos[1],
                         labels=['agent[0]', 'agent[1]'])

    try:
        convergence.plot(agent_pos[0], agent_pos[1], labels=['agent[0]'])
    except:
        convergence.plot(agent_pos[0], agent_pos[1])
