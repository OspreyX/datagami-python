[![Build Status](https://travis-ci.org/datagami/datagami-python.svg?branch=master)](https://travis-ci.org/datagami/datagami-python)

Datagami API for Python
=======================

Basic wrapper to access the Datagami machine learning API.  There are three main methods:

*  upload - send data to the Datagami server
*  train - choose a model and train it on your data
*  predict - make a prediction using your trained model

For some types of model one or more of these steps are combined in convenience methods:

* forecast1D - train a model and return a prediction
* auto1D - train many models and retun a list ordered by accuracy of their forecasts 

Simple Example
==============

You can forecast a 'noisy sine' curve with a single line:

```python
import numpy as np
import datagami

t = np.arange(100)
y = 0.1*t + np.sin(t/3) + np.random.normal(t.size)

f = datagami.forecast1D(y[-10:], kernel='SE', n=10)
```

And we can use ggplot to display the results:
```python
import pandas
from ggplot import *

df = pandas.DataFrame({'t': t, 'y': y, 'fit': f['fit']})

ggplot(aes(x='t',y='y'), data=df) + \
    geom_point(color='lightblue') + \
    geom_line(x=t,y=f['fit']) 

```



