import os
import datagami
import json
from numpy import linspace, meshgrid
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata

KEY = os.environ.get('TEST_API_KEY')
SECRET = os.environ.get('TEST_API_SECRET')


def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi)
    X, Y = meshgrid(xi, yi)
    return X, Y, Z


def load_data():

    # Data from Stanford Machine Learning course exercise 8
    with open('classification_nonlinear.json', 'r') as f:
        return json.load(f)

data = load_data()


def draw_data():

    plt.scatter(data['x'], data['y'], c=data['z'], alpha=0.5)
    plt.show()


def draw_model():

    d = datagami.Datagami(KEY, SECRET)

    # Upload training data
    data_key = d.upload_data(data)

    # Train model
    classification_data = d.classification_train(
        data_key,
        'z',
        num_trees=1000,
        depth=3,
        cv=2
    )

    isp = classification_data['in_sample_probabilities']

    plt.contourf(*grid(data['x'], data['y'], isp, 300, 300), alpha=0.5)
    plt.scatter(data['x'], data['y'], c=data['z'])

    plt.show()

    return classification_data


def draw_prediction(classification_data):

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

    d = datagami.Datagami(KEY, SECRET)

    # Upload new data
    new_data_key = d.upload_data(new_data)

    # Make prediction
    model_key = classification_data['model_key']
    prediction = d.classification_predict(model_key, new_data_key)

    new_x = [i['x'] for i in new_data]
    new_y = [i['y'] for i in new_data]

    isp = classification_data['in_sample_probabilities']
    plt.contourf(*grid(data['x'], data['y'], isp, 300, 300), alpha=0.3)

    prob = prediction['predicted_probabilities']
    plt.scatter(new_x, new_y, c=prob, s=40)
    plt.show()


if __name__ == '__main__':
    draw_data()
    model = draw_model()
    draw_prediction(model['model_key'])
