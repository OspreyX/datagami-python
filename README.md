[![Build Status](https://travis-ci.org/datagami/datagami-python.svg?branch=master)](https://travis-ci.org/datagami/datagami-python)

Datagami API for Python
=======================

Basic wrapper to access the Datagami machine learning API.  There are three main methods:

*  upload_data - send data to the Datagami server
*  train - choose a model and train it on your data
*  predict - make a prediction using your trained model

For some types of model one or more of these steps are combined in convenience methods:

* forecast1D - train a model and return a prediction
* auto1D - train many models and retun a list ordered by accuracy of their forecasts 

## Examples

### Linear forecast

```python
data = range(100)
forecast = datagami.forecast1D(data, '<key>', '<secret>', kernel='Lin', steps_ahead=10)
print forecast['predicted']
```

Outputs:

```
[99.9999999999655,
 100.999999999942,
 101.999999999986,
 102.999999999945,
 103.999999999971,
 104.999999999953,
 105.999999999976,
 106.999999999961,
 107.999999999962,
 108.999999999974]
```

### Noisy sine curve

```python
t = np.arange(100)
sine = 0.1*t + np.sin(t/3) + np.random.normal(t.size)
forecast = datagami.forecast1D(list(sine), '<key>', '<secret>', kernel='SE', steps_ahead=10)
print forecast
```

TODO add graph