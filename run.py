from opytimizer.utils.history import History

h = History()
h.load('snapshot_iter_100.pkl')

print(h.best_agent[-1])