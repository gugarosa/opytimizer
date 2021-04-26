from opytimizer.utils import attribute


def test_attribute_rgetattr():
    class Object:
        pass

    new_attribute = Object()
    new_attribute.x = Object()
    new_attribute.x.a = 'xa'

    assert attribute.rgetattr(new_attribute, 'x.a') == 'xa'


def test_attribute_rsetattr():
    class Object:
        pass

    new_attribute = Object()
    new_attribute.x = Object()

    attribute.rsetattr(new_attribute, 'x.a', 'xa')

    assert new_attribute.x.a == 'xa'
