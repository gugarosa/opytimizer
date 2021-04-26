import opytimizer.utils.attribute as a


class AttributeItem:
    """AttributeItem class is just a mock-up for adding attributes.

    """

    def __init__(self):
        """Initialization method.

        """

        pass


class Attribute:
    """Attribute class is just a mock-up for showing the power of `attributes` module.

    """

    def __init__(self):
        """Initialization method.

        """

        # One-level property
        self.item1 = 'item1'

        # Nested property
        self.item2 = AttributeItem()
        self.item2.arg1 = 'arg1'
        self.item2.arg2 = 'arg2'


# Instantiates the Attribute class
attr = Attribute()

# Gathers one-level and nested attributes
item1 = a.rgetattr(attr, 'item1')
item2 = a.rgetattr(attr, 'item2')
item2_arg1 = a.rgetattr(attr, 'item2.arg1')
item2_arg2 = a.rgetattr(attr, 'item2.arg2')
print(item1, item2, item2_arg1, item2_arg2)

# Sets one-level and nested attributes
a.rsetattr(attr, 'item1', '_item1')
a.rsetattr(attr, 'item2.arg1', '_arg1')
a.rsetattr(attr, 'item2.arg2', '_arg2')
print(attr.item1, attr.item2, attr.item2.arg1, attr.item2.arg2)
