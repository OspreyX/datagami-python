
import numpy as np
import datagami
import pprint

## TS 1D tests
x = np.arange(100)
y = np.sin(x*np.pi/10)*x*0.1 + np.random.normal(size=100)


print '============================================'
print '----  1D  forecast 10 -----'
dg = datagami.forecast1D(y, 'RQ', 10)

dg.pop("fit", None)
dg.pop("fit_variance", None)
pprint.pprint(dg, indent=4)


print '============================================'
print '----  1D  forecast 1 -----'
dg = datagami.forecast1D(y, 'RQ', 1)

dg.pop("fit", None)
dg.pop("fit_variance", None)
pprint.pprint(dg, indent=4)



# res = datagami.auto1D(y, kl=["SE","RQ","SE + RQ","Lin*SE"], n=10, url='local')

# print '---- res 1D auto -----'
# pprint.pprint(res, indent=4)


## TS nD tests
x = np.arange(100)
y1 = 0.2*x + np.sin(x/20) + np.random.normal(np.zeros(100))
y2 = 0.3*y1 + np.random.normal(np.zeros(100)) 
y3 = 0.1*y1**2 + np.random.normal(np.zeros(100)) 
y4 = 0.5*y1**3 + np.random.normal(np.zeros(100)) 

dat_train = {'a': y1[:-2], 'b': y2[:-2], 'd': y4[1:-1], 'c': y3[1:-1]}
dat_test1 = {'a': y1[-2:-1], 'b': y2[-2:-1]} 
dat_test2 = {'a': y1[-2:], 'b': y2[-2:]}


print '============================================'
print '---- nD setup -----'
nD = datagami.TimeSeriesND(dat_train)

pprint.pprint(nD)

print '============================================'
print '---- nD train  -----'
res1 = nD.train(columns_to_predict=['c','d'], kernel='RQ')

pprint.pprint(res1, indent=4)

print '---- nD predict > 1 -----'
res2 = nD.predict(dat_test2)

res2.pop('fit', None)
res2.pop('fit_variance', None)
pprint.pprint(res2, indent=4)

print '---- nD predict 1 -----'
res3 = nD.predict(dat_test1)

res3.pop('fit', None)
res3.pop('fit_variance', None)
pprint.pprint(res3, indent=4)

print '============================================'
print '----- nD setup: single target variable -----'
dat_train = {'a': y1[:-2], 'b': y2[:-2], 'd': y4[:-2], 'c': y3[1:-1]}
dat_test1 = {'a': y1[-2:-1], 'b': y2[-2:-1], 'd': y4[-2:-1]} 
dat_test2 = {'a': y1[-2:], 'b': y2[-2:], 'd': y4[-2:]}
nD = datagami.TimeSeriesND(dat_train, url='local')

print '---- nD train -----'
res1 = nD.train(columns_to_predict='c', kernel='RQ')

pprint.pprint(res1, indent=4)

print '---- nD predict > 1 -----'
res2 = nD.predict(dat_test2)

res2.pop('fit', None)
res2.pop('fit_variance', None)
pprint.pprint(res2, indent=4)

print '---- nD predict 1 -----'
res3 = nD.predict(dat_test1)

res3.pop('fit', None)
res3.pop('fit_variance', None)
pprint.pprint(res3, indent=4)


