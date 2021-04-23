from opytimizer.core.agent import Agent

a = Agent(2, 2, [0,0], [1,1])
a.fill_with_uniform()
print(a.position)

a.fill_with_static([1.7, 2.5])
print(a.position)

a.fill_with_binary()
print(a.position)