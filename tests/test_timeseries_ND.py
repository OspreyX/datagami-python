import os
import json
import datagami
from .utils import assert_list_almost_equals


KEY = os.environ.get('TEST_API_KEY')
SECRET = os.environ.get('TEST_API_SECRET')


def test_timeseries_nd():

    with open('tests/timeseries_nd_gdp.json', 'r') as f:
        data = json.load(f)

    new_data = {
        "UnRate": [7.8, 7.9, 7.5, 7.5, 7.2, 6.7, 6.7, 6.1],
        "DTW": [72.75, 73.54, 76.38, 77.72, 75.37, 76.44, 76.86, 75.91],
        "CPI": [231.09, 231.1, 232.07, 232.86, 233.74, 234.59, 235.64, 237.69],
        "SP500": [1440.67, 1426.19, 1569.19, 1606.28, 1681.55, 1848.36, 1872.34, 1960.23],
        "TRADE": [-57431.2, -46386.4, -45093, -50135.3, -63453.3, -50509.3, -51441.3, -61801.3],
        "HOUST": [847, 976, 994, 831, 863, 1034, 950, 1001],
        "INDPRO": [97.39, 98.36, 99.49, 99.61, 100.72, 101.56, 103.16, 103.92]
    }

    forecast = datagami.timeseries_ND(
        KEY, SECRET, data, new_data,
        columns_to_predict=['GDP', 'PAYEM'],
        kernel='RQ'
    )

    assert_list_almost_equals(
        forecast['predicted']['GDP'],
        [
            16463.0216110124,
            16559.8050628565,
            16641.3279496213,
            16669.5412788372,
            16713.7911365902,
            16649.098744842,
            16732.478941268,
            16565.0143350847
        ]
    )

    assert_list_almost_equals(
        forecast['predicted']['PAYEM'],
        [
            134692.981149631,
            134686.379840353,
            135302.064812298,
            135453.722991722,
            135878.552431587,
            136320.963205007,
            136488.538500389,
            136841.217309462
        ]
    )
