
import numpy as np
import datagami

x = np.arange(100)
y = np.sin(x*np.pi/10)*x*0.1 + np.random.normal(size=100)

# f = datagami.TimeSeries1D(y).forecast('RQ', 5)

dg = datagami.TimeSeries1D(y)
res = dg.auto(kernel_list=["SE","RQ","SE + RQ","Lin*SE"], out_of_sample_size=10)




