
import numpy
import json
import requests
import time


# TODO: authentication
# ----------------------------------------------------------------------------------------
# parent class
# ----------------------------------------------------------------------------------------

class Datagami:

	def __init__(self, username='demo', token=''):
		'''
		Create a Datagami object which contains connection to API server
		'''
		self.base_url = 'http://localhost:8888'
		self.data_url = self.base_url + '/v1/data'
		self.model_url = self.base_url + '/v1/model'
		self.forecast_url = self.base_url + '/v1/timeseries/1D/forecast'
		self.auto_url = self.base_url + '/v1/timeseries/1D/auto'


	def getData(self):
		'''
		Retrieve previously uploaded data
		'''
		try:
			self.data_key
		except AttributeError:
			raise ValueError('data not defined')
		r = requests.get(self.data_url + '/' + self.data_key)
		r.raise_for_status()
		data = r.json()['data']
		if self.data_type is numpy.ndarray:
			data = numpy.array(data)
		return data

	def getModel(self):
		'''
		Retrieve previously trained model
		'''
		try:
			self.model_key
		except AttributeError:
			raise ValueError('model not defined')
		r = requests.get(self.model_url + '/' + self.model_key)
		r.raise_for_status()
		data = r.json()['data']
		if self.data_type is numpy.ndarray:
			data = numpy.array(data)
		return data


	def poll(self, url):
		'''
		Poll given url until status is SUCCESS
		'''
		counter = 0
		s = 1 			# initial sleep interval
		inc = 0.5  		# amount to increase interval each loop
		while True:
			time.sleep(s)
			r = requests.get(self.base_url + url)
			resp = r.json()
			if resp['status'] == 'SUCCESS':
				break
			elif resp['status'] == 'FAIL':
				print resp
				raise requests.exceptions.RequestException('Sorry, forecast job failed')
			counter += 1
			s += inc
			if s > 5 and s < 20:
				inc = 3
			elif s > 20:
				inc = 5
			if counter > 1000:
				raise requests.exceptions.Timeout('Sorry, forecast job never completed')

		return resp


	def validateArray1D(self, x):
		'''Sanity checks on 1D timeseries. Must be a python list of floats or a 1D numpy array.'''
		if type(x) is numpy.ndarray:
			self.data_type = numpy.ndarray
			if len(x.shape) != 1:
				raise ValueError('Not a 1D arrray')
			return x.tolist()
		elif type(x) is list:
			self.data_type = list
			return map(float, x)
		else:
			raise ValueError('Error: x must be a numpy array or a python list of floats')

	def validateKernel(self, k):
		if type(k) is not str:
			raise ValueError
		return k

	def validatePositiveInteger(self, n):
		if type(n) is not int:
			i = int(n)
			if i < 1:
				raise ValueError
			return i
		else:
			return n

# ----------------------------------------------------------------------------------------
# endpoint specific class
# ----------------------------------------------------------------------------------------


class TimeSeries1D(Datagami):
	'''
	Class to handle all 1D timeseries models. Instances contain details about the API connection
	and references to the uploaded data.  Methods on this object are: getData, getModel, forecast, and auto.

	'''
	def __init__(self, x, username='demo', token=''):
		'''
		Create a TimeSeries1D object from input data x. 
		Currently, x must be a numpy array or a python list of floats.
		'''
		# authentication is handeld in parent's constructor
		Datagami.__init__(self, username, token)

		# sanity check on data array, converts numpy to python list
		y = self.validateArray1D(x)

		# upload timeseries data to the API
		data_json = json.dumps(y)
		r = requests.post(self.data_url, data={'data':data_json})
		r.raise_for_status()
		r_data = r.json()
		if 'data_key' not in r_data:
			raise requests.exceptions.RequestException(r_data)
		self.data_key = r_data['data_key']
		
	
	def forecast(self, kernel='SE', steps_ahead=10):
		'''
		Fits a Gaussian Process model with the supplied kernel to input array x.  
		Returns a dictionary of results: fit, fit_var, pred, pred_var.
		See API documentation for details.
		'''
		# validation
		n = self.validatePositiveInteger(steps_ahead)
		assert(type(kernel) is str or type(kernel) is unicode)

		# post to forecast endpoint
		params_dict = {'data_key': self.data_key, 'kernel':kernel, 'steps_ahead':n }
		r = requests.post(self.forecast_url, data=params_dict)
		r.raise_for_status()

		# poll for results
		r_forecast = r.json()
		result = self.poll(r_forecast['url'])

		self.model_key = result['model_key']

		# clean up object for return to user
		result.pop('job_id', None)
		result.pop('status', None)
		result.pop('data_key', None)
		result.pop('model_key', None)
		result.pop('type', None)
		result.pop('steps_ahead', None)

		# return numpy if input was numpy
		if self.data_type == "numpy":
			for a in ['fit','fit_var','pred','pred_var']:
				result[a] = numpy.array(result[a])

		return result


	def auto(self, kernel_list=None, out_of_sample_size=10):
		'''
		Tries to find the best Gaussian Process model for input array x, by training 
		a collection of models on a reduced data set, x[:-out_of_sample_size] in python notation,
		and ranking them by the accuracy of their predictions on x[:out_of_sample_size].
		Returns a list of dictionaries.
		'''
		# validation
		n = self.validatePositiveInteger(out_of_sample_size)
		assert type(kernel_list) is list, "ValueError: kernel_list must be a python list"
		assert all(type(k) == str for k in kernel_list), "ValueError: kernel_list must be a python list of strings"
		
		params_dict = {'data_key': self.data_key, 'kernel_list': json.dumps(kernel_list), 'oos_window':n }

		print params_dict

		# post to the auto endpoint
		print 'auto_url', self.auto_url
		r = requests.post(self.auto_url, data=params_dict)
		r.raise_for_status()
		
		# poll for results
		r_auto = r.json()
		result = self.poll(r_auto['url'])

		# store references to cloud objects
		# NOTE: we store the meta_key here as this object's model_key 
		self.model_key = result['meta_key']
		self.job_id = result['job_id']
		self.model_keys = result['model_keys']

		# clean up object for return to user
		for k in ['job_id', 'status', 'data_key', 'model_keys', 'meta_key', 'oos_window', 'type']:
			result.pop(k, None)

		# turn results into a sorted list and return numpy if input was numpy
		result_list = []
		for k,v in result.iteritems():
			v['kernel'] = k
			if self.data_type == "numpy":
				for a in ['fit','fit_var','pred','pred_var']:
					v[a] = numpy.array(v[a])
			result_list.append(v)

		result_list.sort(key=lambda x: x['prediction_error'])

		return result_list



# ----------------------------------------------------------------------------------------
#  convenience methods 
# ----------------------------------------------------------------------------------------

# TODO: authentication

def forecast1D(x, k='SE', n=10):
	'''
	Forecast the 1D timeseries x for n steps ahead, using kernel k.
	Currently, x must be a numpy array or a python list of floats.
	'''
	DG = TimeSeries1D(x)
	f = DG.forecast(k,n)
	return f

def auto1D(x, kl=['SE','RQ','SE + RQ'], n=10):
	'''
	Train models with kernels in kl on timeseries x.  Returns a list of models, ordered
	by prediction accuracy on last n values of x.  
	Currently, x must be a numpy array or a python list of floats.
	'''
	DG = TimeSeries1D(x)
	f = DG.auto(kl, n)
	return f




