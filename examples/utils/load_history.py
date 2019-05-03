from opytimizer.utils.history import History

# Creating an empty History object
h = History()

# Declares the file name to be loaded
file_name = ''

# Loading content from pickle file
h.load(file_name)

# Displaying content
h.show()
