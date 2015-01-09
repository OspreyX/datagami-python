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
forecast = datagami.forecast1D(data, '<key>', '<secret>', kernel='SE', steps_ahead=10)
print forecast['predicted']
```

Outputs:

```
[99.9999421603053,
 100.999908715454,
 101.999865731187,
 102.99981145042,
 103.999743922717,
 104.999660939268,
 105.99956004725,
 106.999438486498,
 107.999293221589,
 108.999120848966]
```

### Noisy sine curve

```python
t = np.arange(100)
sine = 0.1*t + np.sin(t/3) + np.random.normal(t.size)
forecast = datagami.forecast1D(list(sine), '<api_key>', '<api_secret>', kernel='SE', steps_ahead=10)
print forecast
```

TODO add graph