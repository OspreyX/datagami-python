
import numpy as np
import datagami
import pprint

# ## TS 1D tests
# x = np.arange(100)
# y = np.sin(x*np.pi/10)*x*0.1 + np.random.normal(size=100)

# # f = datagami.TimeSeries1D(y).forecast('RQ', 5)

# dg = datagami.forecast1D(y, 'RQ', 10, url='local')

# print '---- res 1D -----'
# pprint.pprint(dg, indent=4)

# res = datagami.auto1D(y, kl=["SE","RQ","SE + RQ","Lin*SE"], n=10, url='local')

# print '---- res 1D auto -----'
# pprint.pprint(res, indent=4)


## TS nD tests
x = np.arange(100)
y1 = 0.2*x + np.sin(x/20) + np.random.normal(np.zeros(100))
y2 = 0.3*y1 + np.random.normal(np.zeros(100)) 
y3 = 0.1*y1**2 + np.random.normal(np.zeros(100)) 

dat_train = {'a': y1[:-2], 'b': y2[:-2], 'c': y3[1:-1]}
dat_test1 = {'a': y1[-2:-1], 'b': y2[-2:-1]} 
dat_test2 = {'a': y1[-2:], 'b': y2[-2:]}


print '---- nD setup -----'
nD = datagami.TimeSeriesND(dat_train, url='local')


print '---- nD res1 -----'
res1 = nD.train(columns_to_predict='c', kernel='RQ')

pprint.pprint(res1, indent=4)

print '---- nD res2 -----'
res2 = nD.predict(dat_test2)

pprint.pprint(res2, indent=4)

print '---- nD res3 -----'
res3 = nD.predict(dat_test1)

pprint.pprint(res3, indent=4)



