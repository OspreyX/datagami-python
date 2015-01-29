import os
import datagami

KEY = os.environ.get('TEST_API_KEY')
SECRET = os.environ.get('TEST_API_SECRET')


def test_linear_convenience():

    data = range(100)
    forecast = datagami.forecast1D(KEY, SECRET, data)

    rounded = [int(p + 0.5) for p in forecast['predicted']]
    assert rounded == range(100, 110)
