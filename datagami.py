
import numpy as np
import json
import requests
import time


# need to connect

class Datagami:

	def __init__(self, username='demo', token=''):
		'''
		Create Datagami object which contains connection to API server
		'''
		self.url = 'http://localhost:8888'

	def poll(self, url):
		'''
		Poll given url until status is SUCCESS
		'''
		counter = 0
		s = 1 			# initial sleep interval
		inc = 0.5  		# amount to increase interval each loop
		while True:
			time.sleep(s)
			r = requests.get(self.url + url)
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



	def validateArray(self, x):
		'''Sanity checks on timeseries. Must be a python list of floats or a 1D numpy array.'''
		if type(x) is np.ndarray:
			if len(x.shape) != 1:
				raise ValueError('Not a 1D arrray')
			return x.tolist()
		elif type(x) is list:
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



class TimeSeries1D(Datagami):
	'''
	Class to handle all 1D timeseries models
	'''
	def __init__(self, x, username='demo', token=''):
		Datagami.__init__(self, username, token)
		self.data_url = self.url + '/v1/data'
		self.forecast_url = self.url + '/v1/timeseries/1D/forecast'
		self.auto_url = self.url + '/v1/timeseries/1D/auto'

		# sanity check on data array
		y = self.validateArray(x)

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

		# clean up object for return to user
		result.pop('job_id', None)
		result.pop('status', None)
		result.pop('data_key', None)
		result.pop('model_key', None)
		result.pop('type', None)
		result.pop('steps_ahead', None)

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

		# post to the auto endpoint
		r = requests.post(self.auto_url, data=params_dict)
		r.raise_for_status()
		
		# poll for results
		r_auto = r.json()
		result = self.poll(r_auto['url'])

		# clean up object for return to user
		result.pop('job_id', None)
		result.pop('status', None)
		result.pop('data_key', None)
		result.pop('model_keys', None)
		result.pop('meta_key', None)
		result.pop('oos_window', None)
		result.pop('type', None)

		# turn results into a sorted list
		result_list = []
		for k,v in result.iteritems():
			v['kernel'] = k
			result_list.append(v)

		return result_list

