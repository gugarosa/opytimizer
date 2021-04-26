from opytimizer import Opytimizer

# opt = Opytimizer()
opt = Opytimizer.load('out.pkl')
# print(len(opt.history.time))
opt.start(n_iterations=100)
print(opt.history.time)