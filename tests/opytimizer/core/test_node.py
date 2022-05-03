import numpy as np

from opytimizer.core import node


def test_node():
    new_node = node.Node(name="0", category="FUNCTION")

    print(repr(new_node))
    print(new_node)


def test_node_name():
    new_node = node.Node(name="0", category="FUNCTION")

    assert new_node.name == "0"


def test_node_name_setter():
    try:
        new_node = node.Node(name=0.0, category="FUNCTION")
    except:
        new_node = node.Node(name=0, category="FUNCTION")

    try:
        new_node = node.Node(name=0.0, category="FUNCTION")
    except:
        new_node = node.Node(name="0", category="FUNCTION")

    assert str(new_node.name) == "0"


def test_category():
    new_node = node.Node(name="0", category="FUNCTION")

    assert new_node.category == "FUNCTION"


def test_category_setter():
    try:
        new_node = node.Node(name=0, category="F")
    except:
        new_node = node.Node(name=0, category="FUNCTION")

    assert new_node.category == "FUNCTION"

    try:
        new_node = node.Node(name=0, category="T")
    except:
        new_node = node.Node(name=0, category="TERMINAL", value=np.array(0))

    assert new_node.category == "TERMINAL"


def test_node_value():
    new_node = node.Node(name="0", category="TERMINAL", value=np.array(0))

    assert new_node.value == 0


def test_node_value_setter():
    try:
        new_node = node.Node(name=0, category="TERMINAL", value=0)
    except:
        new_node = node.Node(name=0, category="TERMINAL", value=np.array(0))

    assert new_node.value == 0


def test_node_left():
    new_node = node.Node(name="0", category="FUNCTION")

    assert new_node.left is None


def test_node_left_setter():
    try:
        new_node = node.Node(name=0, category="FUNCTION", left=1)
    except:
        new_node2 = node.Node(name=0, category="TERMINAL", value=np.array(0))

        new_node = node.Node(name=0, category="FUNCTION", left=new_node2)

    assert isinstance(new_node.left, node.Node)


def test_node_right():
    new_node = node.Node(name="0", category="FUNCTION")

    assert new_node.right is None


def test_node_right_setter():
    try:
        new_node = node.Node(name=0, category="FUNCTION", right=1)
    except:
        new_node2 = node.Node(name=0, category="TERMINAL", value=np.array(0))

        new_node = node.Node(name=0, category="FUNCTION", right=new_node2)

    assert isinstance(new_node.right, node.Node)


def test_node_parent():
    new_node = node.Node(name="0", category="FUNCTION")

    assert new_node.parent is None


def test_node_parent_setter():
    try:
        new_node = node.Node(name=0, category="FUNCTION", parent=1)
    except:
        new_node2 = node.Node(name=0, category="TERMINAL", value=np.array(0))

        new_node = node.Node(name=0, category="FUNCTION", parent=new_node2)

    assert isinstance(new_node.parent, node.Node)


def test_node_flag():
    new_node = node.Node(name="0", category="FUNCTION")

    assert new_node.flag is True


def test_node_flag_setter():
    try:
        new_node = node.Node(name=0, category="FUNCTION")
        new_node.flag = 10
    except:
        new_node = node.Node(name=0, category="FUNCTION")

    assert new_node.flag is True


def test_node_min_depth():
    new_node = node.Node(name="0", category="FUNCTION")

    assert new_node.min_depth == 0


def test_node_max_depth():
    new_node = node.Node(name="0", category="FUNCTION")

    assert new_node.max_depth == 0


def test_node_n_leaves():
    new_node = node.Node(name="0", category="FUNCTION")

    assert new_node.n_leaves == 1


def test_node_n_nodes():
    new_node = node.Node(name="0", category="FUNCTION")

    assert new_node.n_nodes == 1


def test_node_position():
    new_node = node.Node(name="0", category="TERMINAL", value=np.array(0))

    assert new_node.position == 0


def test_node_pre_order():
    new_node = node.Node(name="SUM", category="FUNCTION")
    new_node_1 = node.Node(name="1", category="TERMINAL", value=np.array(0))
    new_node_2 = node.Node(name="2", category="TERMINAL", value=np.array(0))

    new_node.left = new_node_1
    new_node.right = new_node_2
    new_node_1.parent = new_node
    new_node_2.parent = new_node

    assert len(new_node.pre_order) == 3


def test_node_post_order():
    new_node = node.Node(name="SUM", category="FUNCTION")
    new_node_1 = node.Node(name="1", category="TERMINAL", value=np.array(0))
    new_node_2 = node.Node(name="2", category="TERMINAL", value=np.array(0))

    new_node.left = new_node_1
    new_node.right = new_node_2
    new_node_1.parent = new_node
    new_node_2.parent = new_node

    assert len(new_node.post_order) == 3


def test_node_find_node():
    new_node = node.Node(name="0", category="TERMINAL", value=np.array(0))

    assert new_node.find_node(0) == (None, True)
    assert new_node.find_node(1) == (None, False)

    new_node = node.Node(name="SUM", category="FUNCTION", value=np.array(0))

    assert new_node.find_node(0) == (None, False)
    assert new_node.find_node(1) == (None, False)


def test_node_evaluate():
    def _create_node(function_type):
        new_node = node.Node(name=function_type, category="FUNCTION")
        new_node_1 = node.Node(name="1", category="TERMINAL", value=np.array(1))
        new_node_2 = node.Node(name="2", category="TERMINAL", value=np.array(1))

        new_node.left = new_node_1
        new_node.right = new_node_2
        new_node_1.parent = new_node
        new_node_2.parent = new_node

        return new_node

    new_node = _create_node("SUM")
    assert node._evaluate(new_node) == 2

    new_node = _create_node("SUB")
    assert node._evaluate(new_node) == 0

    new_node = _create_node("MUL")
    assert node._evaluate(new_node) == 1

    new_node = _create_node("DIV")
    assert node._evaluate(new_node) == 1

    new_node = _create_node("EXP")
    assert np.round(node._evaluate(new_node)) == 3

    new_node = _create_node("SQRT")
    assert node._evaluate(new_node) == 1

    new_node = _create_node("LOG")
    assert node._evaluate(new_node) == 0

    new_node = _create_node("ABS")
    assert node._evaluate(new_node) == 1

    new_node = _create_node("SIN")
    assert np.round(node._evaluate(new_node)) == 1

    new_node = _create_node("COS")
    assert np.round(node._evaluate(new_node)) == 1


def test_node_properties():
    new_node = node.Node(name="SUM", category="FUNCTION")
    new_node_1 = node.Node(name="1", category="TERMINAL", value=np.array(0))
    new_node_2 = node.Node(name="2", category="TERMINAL", value=np.array(0))

    new_node.left = new_node_1
    new_node.right = new_node_2
    new_node_1.parent = new_node
    new_node_2.parent = new_node

    assert isinstance(node._properties(new_node), dict)
    assert print(new_node) is None
