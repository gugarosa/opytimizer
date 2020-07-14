from opytimizer.utils import history
from opytimizer.visualization import convergence


def test_convergence_plot():
    new_history = history.History()

    new_history.load('models/test.pkl')

    agents = new_history.get(key='agents', index=(0, 0))

    try:
        convergence.plot(agents[0], agents[1], labels=1)
    except:
        convergence.plot(agents[0], agents[1], labels=['agent[0]', 'agent[1]'])

    try:
        convergence.plot(agents[0], agents[1], labels=['agent[0]'])
    except:
        convergence.plot(agents[0], agents[1])
