# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

from opytimizer.utils.history import History

h = History()

h.dump(x=1)
h.dump(x=2)
h.dump(x=3)

# Custom values, including lists and dictionaries, are appended without copying
h.dump(y=[1])

print(h.x)
print(h.y)
