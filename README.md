
Datagami API for Python
=======================

Basic wrapper to access the Datagami machine learning API.  There are three main methods:

*  upload - send data to the Datagami server
*  train - choose a model and train it on your data
*  predict - make a prediction using your trained model

For some types of model one or more of these steps are combined in convenience methods:

* forecast - train a model and return a prediction
* auto - train many models and retun a list ordered by accuracy of their forecasts 

Simple Example
==============

Here we forecast a 'noisy sine' curve:

```python
	import numpy as np
	import matplotlib as mp
	import datagami

	t = np.arange(1:100)
	x = 0.1*t + np.sin(t/10) * np.random(t.size)

	dg = datagami()
	f = dg.forecast(x)

	mp.plot(t,x,f)
```



