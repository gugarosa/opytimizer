import numpy as np

from opytimizer.core import node


def test_node():
    new_node = node.Node(name='0', type='FUNCTION')

    print(repr(new_node))
    print(new_node)


def test_node_name():
    new_node = node.Node(name='0', type='FUNCTION')

    assert new_node.name == '0'


def test_node_name_setter():
    try:
        new_node = node.Node(name=0.0, type='FUNCTION')
    except:
        new_node = node.Node(name=0, type='FUNCTION')

    try:
        new_node = node.Node(name=0.0, type='FUNCTION')
    except:
        new_node = node.Node(name='0', type='FUNCTION')

    assert str(new_node.name) == '0'


def test_node_type():
    new_node = node.Node(name='0', type='FUNCTION')

    assert new_node.type == 'FUNCTION'


def test_node_type_setter():
    try:
        new_node = node.Node(name=0, type='F')
    except:
        new_node = node.Node(name=0, type='FUNCTION')

    assert new_node.type == 'FUNCTION'

    try:
        new_node = node.Node(name=0, type='T')
    except:
        new_node = node.Node(name=0, type='TERMINAL', value=np.array(0))

    assert new_node.type == 'TERMINAL'


def test_node_value():
    new_node = node.Node(name='0', type='TERMINAL', value=np.array(0))

    assert new_node.value == 0


def test_node_value_setter():
    try:
        new_node = node.Node(name=0, type='TERMINAL', value=0)
    except:
        new_node = node.Node(name=0, type='TERMINAL', value=np.array(0))

    assert new_node.value == 0


def test_node_left():
    new_node = node.Node(name='0', type='FUNCTION')

    assert new_node.left == None


def test_node_left_setter():
    try:
        new_node = node.Node(name=0, type='FUNCTION', left=1)
    except:
        new_node2 = node.Node(name=0, type='TERMINAL', value=np.array(0))

        new_node = node.Node(name=0, type='FUNCTION', left=new_node2)

    assert isinstance(new_node.left, node.Node)


def test_node_right():
    new_node = node.Node(name='0', type='FUNCTION')

    assert new_node.right == None


def test_node_right_setter():
    try:
        new_node = node.Node(name=0, type='FUNCTION', right=1)
    except:
        new_node2 = node.Node(name=0, type='TERMINAL', value=np.array(0))

        new_node = node.Node(name=0, type='FUNCTION', right=new_node2)

    assert isinstance(new_node.right, node.Node)


def test_node_parent():
    new_node = node.Node(name='0', type='FUNCTION')

    assert new_node.parent == None


def test_node_parent_setter():
    try:
        new_node = node.Node(name=0, type='FUNCTION', parent=1)
    except:
        new_node2 = node.Node(name=0, type='TERMINAL', value=np.array(0))

        new_node = node.Node(name=0, type='FUNCTION', parent=new_node2)

    assert isinstance(new_node.parent, node.Node)


def test_node_flag():
    new_node = node.Node(name='0', type='FUNCTION')

    assert new_node.flag == True


def test_node_flag_setter():
    try:
        new_node = node.Node(name=0, type='FUNCTION')
        new_node.flag = 10
    except:
        new_node = node.Node(name=0, type='FUNCTION')

    assert new_node.flag == True


def test_node_min_depth():
    new_node = node.Node(name='0', type='FUNCTION')

    assert new_node.min_depth == 0


def test_node_max_depth():
    new_node = node.Node(name='0', type='FUNCTION')

    assert new_node.max_depth == 0


def test_node_n_leaves():
    new_node = node.Node(name='0', type='FUNCTION')

    assert new_node.n_leaves == 1


def test_node_n_nodes():
    new_node = node.Node(name='0', type='FUNCTION')

    assert new_node.n_nodes == 1


def test_node_position():
    new_node = node.Node(name='0', type='TERMINAL', value=np.array(0))

    assert new_node.position == 0


def test_node_pre_order():
    new_node = node.Node(name='0', type='FUNCTION')

    assert len(new_node.pre_order) == 1


def test_node_post_order():
    new_node = node.Node(name='0', type='FUNCTION')

    assert len(new_node.post_order) == 1


def test_node_find_node():
    new_node = node.Node(name='0', type='TERMINAL', value=np.array(0))

    assert new_node.find_node(0) == (None, True)
    assert new_node.find_node(1) == (None, False)


def test_node_properties():
    new_node = node.Node(name='0', type='FUNCTION')

    assert isinstance(node._properties(new_node), dict)
