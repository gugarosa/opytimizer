from opytimizer.utils import exception


def test_error():
    new_exception = exception.Error('Error', 'error')

    try:
        raise new_exception
    except exception.Error:
        pass


def test_argument_error():
    new_exception = exception.ArgumentError('error')

    try:
        raise new_exception
    except exception.ArgumentError:
        pass


def test_build_error():
    new_exception = exception.BuildError('error')

    try:
        raise new_exception
    except exception.BuildError:
        pass


def test_size_error():
    new_exception = exception.SizeError('error')

    try:
        raise new_exception
    except exception.SizeError:
        pass


def test_type_error():
    new_exception = exception.TypeError('error')

    try:
        raise new_exception
    except exception.TypeError:
        pass


def test_value_error():
    new_exception = exception.ValueError('error')

    try:
        raise new_exception
    except exception.ValueError:
        pass
