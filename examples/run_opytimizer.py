from opytimizer import Opytimizer
from opytimizer.core.space import Space
from opytimizer.functions.internal import Internal
from opytimizer.optimizers.pso import PSO


def test(x):
    return x + 1

# Building up Space
s = Space(n_agents=2)
s.build(n_variables=5, n_dimensions=1)

# Building up PSO's optimizer
p = PSO()

# Building up Internal function
f = Internal()
f.build(test)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

o.evaluate()
