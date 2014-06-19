
import numpy as np
import datagami

x = np.arange(100)
y = np.sin(x*np.pi/10)*x*0.1 + np.random.normal(size=100)

# f = datagami.TimeSeries1D(y).forecast('RQ', 5)

# g = datagami.TimeSeries1D(y).auto( ["SE","RQ"], 5)




# {	u'RQ': {
# 			u'BIC': 184.05471171906336,
# 			u'hyper-parameters': [1.9453345439465, 5.55529550232035, 50, 0.358734913234334],
# 			u'model_key': u'0beecd6da46d5b30ccb6abaa684a0e1938eb3929',
# 			u'prediction_error': -0.31452597853041625
# 		},
#  	u'SE': {
#  			u'BIC': 179.4007901480952,
# 			u'hyper-parameters': [1.95451767657363, 5.60122656240207, 0.360142386221155],
# 			u'model_key': u'7152439a56a896bc3e6f884983285a80002fc1d1',
# 			u'prediction_error': -0.3209356338899173
# 		},
#  	u'data_key': u'5782f74de7d7553b856d25539e759bc5227a5403',
#  	u'job_id': u'19ea95ac-98d5-4ed2-b11e-8e1ae847482c',
#  	u'meta_key': u'ec262c91cd2a7b8693c4bd1a68e4a4b6c53598fa',
#  	u'model_keys': [u'7152439a56a896bc3e6f884983285a80002fc1d1', u'0beecd6da46d5b30ccb6abaa684a0e1938eb3929'],
#  	u'oos_window': 5,
#  	u'status': u'SUCCESS'
#  }
