from opytimizer.utils.history import History

# Instantiates the History
h = History()

# Dumps a variable (it will be converted into a list)
h.dump(x=1)
h.dump(x=2)
h.dump(x=3)

# Any variable will be converted into a list
# Even lists, dictionaries, etc
h.dump(y=[1])

# Access the variables
print(h.x)
print(h.y)
