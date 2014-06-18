
import numpy as np
import datagami

x = np.arange(100)
y = np.sin(x*np.pi/10)*x*0.1 + np.random.normal(size=100)

dgts = datagami.TimeSeries1D()

# f = dgts.forecast(y)

# g = dgts.auto(y[1:50],None,5)
