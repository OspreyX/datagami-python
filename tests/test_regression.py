import os
import datagami
import json

KEY = os.environ.get('TEST_API_KEY')
SECRET = os.environ.get('TEST_API_SECRET')


with open('tests/regression_house_prices.json', 'r') as f:
    data = json.load(f)

new_data = {
    'living area': 1650,
    'bedrooms': 3
}


def test_regression_train():

    d = datagami.Datagami(KEY, SECRET)
    data_key = d.upload_data(data)
    regression_data = d.regression_train(
        data_key,
        'price',
        num_trees=1000,
        depth=1,
        cv=0
    )

    model_key = regression_data['model_key']

    new_data_key = d.upload_data(new_data)
    prediction = d.regression_predict(model_key, new_data_key)

    predicted_house_price = prediction['predicted'][0]
    print 'house price prediction:', predicted_house_price
    assert 290000 < predicted_house_price < 300000


def test_regression_convenience():

    training_params = {
        'num_trees': 1000,
        'depth': 1,
        'cv': 0
    }

    prediction = datagami.regression(KEY, SECRET, data, new_data, 'price', **training_params)

    predicted_house_price = prediction['predicted'][0]
    print 'house price prediction:', predicted_house_price
    assert 290000 < predicted_house_price < 300000

