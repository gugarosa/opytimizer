from opytimizer.core.function import Function

f = Function(type='internal')

f.build('x_0 + x1')

print(f.function)