from opytimizer.utils.history import History

# File name to be loaded
file_name = ''

# Creating an empty History object
h = History()

# Loading history from pickle file
h.load(file_name)

# Displaying content
print(h)
