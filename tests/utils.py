from nose.tools import assert_less_equal


def almost_equal_rel(a, b, tol=1e-8):
    """ Relative almost equal - good for comparing large floats """
    assert_less_equal(abs(a - b), max(abs(a), abs(b)) * tol)


def assert_list_almost_equals(actual, expected):
    try:
        for v1, v2 in zip(actual, expected):
            almost_equal_rel(v1, v2)
    except AssertionError:
        print 'Lists not equal'
        print actual
        print expected
        raise
