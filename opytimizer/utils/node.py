class Node():
    """
    """

    def __init__(self, id, name, type, right=None, left=None, parent=None):
        """Initialization method.

        Args:
            
        """

        #
        self.id = id

        #
        self.name = name

        #
        self.type = type

        #
        self.right = right

        #
        self.left = left

        #
        self.parent = parent

    def show(self):
        """
        """

        print(f'Node: id = {self.id} | name = {self.name} | type = {self.type} | left = {self.left} | right = {self.right} | parent = {self.parent}.')