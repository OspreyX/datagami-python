import datagami

API_URL = ''
KEY = ''
SECRET = ''


def test_linear():
    data = range(100)
    ts = datagami.TimeSeries1D(data, KEY, SECRET, url=API_URL)

    result = ts.forecast('SE', 10)

    rounded = [int(p + 0.5) for p in result['predicted']]
    assert rounded == range(100, 110)


def test_linear_convenience():

    data = range(100)
    result = datagami.forecast1D(data, KEY, SECRET, k='SE', n=10, url=API_URL)

    rounded = [int(p + 0.5) for p in result['predicted']]
    assert rounded == range(100, 110)
