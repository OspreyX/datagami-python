
import numpy as np
import datagami
import pprint

x = np.arange(100)
y = np.sin(x*np.pi/10)*x*0.1 + np.random.normal(size=100)

# f = datagami.TimeSeries1D(y).forecast('RQ', 5)

dg = datagami.forecast1D(y, 'RQ', 10)

pprint.pprint(dg, indent=4)

res = datagami.auto1D(y, kl=["SE","RQ","SE + RQ","Lin*SE"], n=10)

pprint.pprint(res, indent=4)




