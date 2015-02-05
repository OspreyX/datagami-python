import os
import datagami
import json

KEY = os.environ.get('TEST_API_KEY')
SECRET = os.environ.get('TEST_API_SECRET')

# Data from Stanford Machine Learning course exercise 8
with open('tests/classification_nonlinear.json', 'r') as f:
    data = json.load(f)

new_data = [
    {'x': 0.1, 'y': 0.8},
    {'x': 0.2, 'y': 0.8},
    {'x': 0.3, 'y': 0.8},
    {'x': 0.4, 'y': 0.8},
    {'x': 0.5, 'y': 0.8},
    {'x': 0.6, 'y': 0.8},
    {'x': 0.7, 'y': 0.8},
    {'x': 0.8, 'y': 0.8},
    {'x': 0.9, 'y': 0.8}
]


def test_classification_convenience():

    training_params = {
        'num_trees': 1000,
        'depth': 3,
        'cv': 2,
    }

    prediction = datagami.classification(KEY, SECRET, data, new_data, 'z', **training_params)

    assert prediction['predicted_classes'] == [1, 1, 0, 0, 1, 0, 0, 1, 1]
