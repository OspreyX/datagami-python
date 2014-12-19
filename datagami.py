
import numpy
import json
import requests
import time
import logging

VERSION = '1.0.3'

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

# TODO: authentication
# ----------------------------------------------------------------------------------------
# parent class
# ----------------------------------------------------------------------------------------


class Datagami(object):

    def __init__(self, username='demo', token='', url=None):
        '''
        Create a Datagami object which contains connection to API server
        '''
        # sanity checks on url
        if url is None:
            self.base_url = 'http://beta.api.datagami.net'
        elif url == 'local':
            self.base_url = "http://localhost:8888"
        else:
            raise ValueError("unknown url: %s" % url)

        self.data_url = self.base_url + '/v1/data'
        self.model_url = self.base_url + '/v1/model'
        self.forecast1D_url = self.base_url + '/v1/timeseries/1D/forecast'
        self.auto1D_url = self.base_url + '/v1/timeseries/1D/auto'
        self.ts_trainND_url = self.base_url + '/v1/timeseries/nD/train'
        self.ts_predictND_url = self.base_url + '/v1/timeseries/nD/predict'

        # for handling numpy arrays cleanly
        self.data_type = None

    def get_data(self):
        '''
        Retrieve previously uploaded data
        '''
        try:
            self.data_key
        except AttributeError:
            raise NameError('data not defined')
        r = requests.get(self.data_url + '/' + self.data_key)
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
        s = 1           # initial sleep interval
        inc = 0.5       # amount to increase interval each loop
        while True:
            r = requests.get(self.base_url + url)
            resp = r.json()
            if resp['status'] == 'SUCCESS':
                job_done = True
                break
            elif resp['status'] == 'FAILURE':
                job_done = False
                raise requests.exceptions.RequestException('Sorry, job failed')
            time.sleep(s)
            counter += 1
            s += inc
            if s > 5 and s < 20:
                inc = 3
            elif s > 20:
                inc = 5
            if counter > 1000:
                raise requests.exceptions.Timeout('Sorry, job timedout')

        return job_done

    def poll_retrieve(self, job):
        '''
        Takes object returned by initial job submission, polls the model endpoint
        until the calculation has finished, then retrieves the results object
        '''
        # store some relevant details
        self.model_key = job['model_key']

        # poll for results
        job_flag = self.poll(job['model_url'])
        if not job_flag:
            raise ValueError("Polling for model %s failed" % self.model_key)

        # get model details
        r = requests.get(self.base_url + job['model_url'])
        r.raise_for_status()
        result = r.json()

        return result

    def validateArray1D(self, x):
        '''Sanity checks on 1D timeseries. Must be a python list of floats or a numpy array.'''
        if type(x) is numpy.ndarray:
            self.data_type = numpy.ndarray
            if len(x.shape) != 1:
                raise ValueError('Not a 1D arrray')
            return x.tolist()
        elif type(x) is list:
            self.data_type = list
            return map(float, x)
        else:
            raise ValueError('x must be a numpy array or a python list of floats')

    def validateArrayND(self, x):
        '''Sanity checks on nD timeseries. Must be a dictionary of lists or numpy arrays.'''
        if type(x) is not dict:
            raise ValueError("x must be a dictionary. Found %s." % type(x))
        # manage possible types inside x
        for k, v in x.iteritems():
            # check column names
            if type(k) is not str:
                raise ValueError('x must be a dictionary with keys of type str.  Found %s.' % type(k))
            # check consistency amongst columns
            if type(v) is numpy.ndarray and self.data_type is None:
                self.data_type = numpy.ndarray
            elif type(v) is list and self.data_type is None:
                self.data_type = list
            elif self.data_type is not None:
                if type(v) != self.data_type:
                    ValueError('All columns must be the same type.  Found %s and %s.' % (self.data_type, type(v)))
            # turn columns into lists of floats
            if type(v) is numpy.ndarray:
                if len(v.shape) != 1:
                    raise ValueError('Column in data dictionary can only be a 1D arrray, found %s' % str(v.shape))
                x[k] = v.tolist()
            elif type(v) is list:
                self.data_type = list
                try:
                    x[k] = map(float, v)
                except ValueError:
                    raise ValueError('Could not convert list element to float, in column %s' % k)
            else:
                raise ValueError('Columns must be numpy arrays or python lists of floats, found %s' % type(v))

        return x

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
    and references to the uploaded data.  Methods on this object are: get_data, forecast, and auto.

    '''

    def __init__(self, x, username='demo', token='', url=None):
        '''
        Create a TimeSeries1D object from input data x.
        Currently, x must be a numpy array or a python list of floats.
        '''
        # authentication is handled in parent's constructor
        Datagami.__init__(self, username, token, url)

        # sanity check on data array, converts numpy to python list
        self.data_type = None
        y = self.validateArray1D(x)

        # upload timeseries data to the API
        data_json = json.dumps(y)
        r = requests.post(self.data_url, data={'data': data_json})
        r.raise_for_status()
        r_data = r.json()
        if 'data_key' not in r_data:
            raise requests.exceptions.RequestException(r_data)
        self.data_key = r_data['data_key']

    def forecast(self, kernel='SE', steps_ahead=10):
        '''
        Fits a Gaussian Process model with the supplied kernel to input array x.
        Returns a dictionary of results: fit, fit_variance, predicted, predicted_variance.
        See API documentation for details.
        '''
        # validation
        n = self.validatePositiveInteger(steps_ahead)
        assert(type(kernel) is str or type(kernel) is unicode)

        # post to forecast endpoint
        params_dict = {'data_key': self.data_key, 'kernel': kernel, 'steps_ahead': n}
        r = requests.post(self.forecast1D_url, data=params_dict)
        r.raise_for_status()

        # get results from server, synchronous, ie poll and wait
        result = self.poll_retrieve(r.json())

        # clean up object for return to user
        # result.pop('job_id', None)
        result.pop('status', None)
        result.pop('data_key', None)
        # result.pop('model_key', None)
        result.pop('type', None)
        result.pop('steps_ahead', None)

        # return numpy if input was numpy
        if self.data_type is numpy.array:
            for a in ['fit', 'fit_variance', 'predicted', 'predicted_variance']:
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

        params_dict = {'data_key': self.data_key, 'kernel_list': json.dumps(kernel_list), 'oos_window': n}

        # post to the auto endpoint
        r = requests.post(self.auto1D_url, data=params_dict)
        r.raise_for_status()

        # get results from server, synchronous, ie poll and wait
        result = self.poll_retrieve(r.json())

        # store references to server-side objects
        self.model_keys = result['model_keys']
        self.meta_key = result['meta_key']

        # return numpy if input was numpy
        result_dict = result['models']
        for k, v in result_dict.iteritems():
            for x in ['kernel', 'model_key', 'data_key', 'status', 'type']:
                v.pop(x, None)
            if self.data_type is numpy.array:
                for a in ['fit', 'fit_variance', 'predicted', 'predicted_variance']:
                    v[a] = numpy.array(v[a])

        return result_dict

# ----------------------------------------------------------------------------------------


class TimeSeriesND(Datagami):

    '''
    Class to handle all ND timeseries models. Instances contain details about the API connection
    and references to the uploaded data.  Methods on this object are: get_data, getModel, train, and predict.

    '''

    def __init__(self, x, username='demo', token='', url=None):
        '''
        Create a TimeSeriesND object from input data x.  Note, x must be a
        dictionary of numpy or list of floats, ie keys are names and values are column vectors.
        '''
        # authentication is handeld in parent's constructor
        Datagami.__init__(self, username, token, url)

        # sanity check on data array, converts numpy to python list
        y = self.validateArrayND(x)

        # upload timeseries data to the API
        data_json = json.dumps(y)
        r = requests.post(self.data_url, data={'data': data_json})
        r.raise_for_status()
        r_data = r.json()

        if 'data_key' not in r_data:
            raise requests.exceptions.RequestException(r_data)

        self.data_key = r_data['data_key']

    def train(self, columns_to_predict, kernel='SE'):
        '''
        Fits a Gaussian Process model with the supplied kernel to input array x.
        The variables to predict are given by name in the argumet columns_to_predict.
        Returns a model_keydictionary of results: fit, fit_variance, predicted, predicted_variance.
        See API documentation for details.
        '''
        # validation
        if type(kernel) not in (str, unicode):
            raise ValueError("kernel must be text (string or unicode).  Found %s" % type(kernel))

        if type(columns_to_predict) in (str, unicode):
            columns_to_predict = [columns_to_predict]
        elif type(columns_to_predict) not in (tuple, list):
            raise ValueError("columns_to_predict must be a single text or tuple/list of text")
        elif type(columns_to_predict) in (tuple, list):
            if not all([type(s) in (str, unicode) for s in columns_to_predict]):
                raise ValueError("columns_to_predict must contain only text or unicode")
        columns_to_predict = json.dumps(columns_to_predict)

        # post to train endpoint
        params_dict = {'data_key': self.data_key, 'kernel': kernel, 'columns_to_predict': columns_to_predict}
        r = requests.post(self.ts_trainND_url, data=params_dict)
        r.raise_for_status()

        # get results from server, synchronous, ie poll and wait
        result = self.poll_retrieve(r.json())

        # store some values
        self.model_key = result['model_key']

        # clean up object for return to user
        # result.pop('job_id', None)
        result.pop('status', None)
        result.pop('data_key', None)
        result.pop('model_key', None)
        result.pop('type', None)
        result.pop('steps_ahead', None)

        # return numpy if input was numpy
        if self.data_type is numpy.array:
            for a in ['fit', 'fit_variance', 'predicted', 'predicted_variance']:
                result[a] = numpy.array(result[a])

        return result

    def predict(self, newdata):
        '''
        Fits a Gaussian Process model with the supplied kernel to input array x.
        The variables to predict are given by name in the argumet columns_to_predict.
        Returns a model_keydictionary of results: fit, fit_var, pred, pred_var.
        See API documentation for details.
        '''
        # validation
        y = self.validateArrayND(newdata)

        # upoload newdata
        newdata_json = json.dumps(y)
        r = requests.post(self.data_url, data={'data': newdata_json})
        r.raise_for_status()
        r_data = r.json()
        if 'data_key' not in r_data:
            raise requests.exceptions.RequestException(r_data)
        self.newdata_key = r_data['data_key']

        # post to train endpoint
        params_dict = {'new_data_key': self.newdata_key, 'model_key': self.model_key}
        r = requests.post(self.ts_predictND_url, data=params_dict)
        r.raise_for_status()

        # get results from server, synchronous, ie poll and wait
        result = self.poll_retrieve(r.json())

        # store some values
        self.model_key = result['model_key']

        # clean up object for return to user
        # result.pop('job_id', None)
        result.pop('status', None)
        result.pop('data_key', None)
        result.pop('new_data_key', None)
        result.pop('model_key', None)
        result.pop('type', None)
        result.pop('message', None)

        # return numpy if input was numpy
        if self.data_type is numpy.array:
            for a in ['fit', 'fit_variance', 'predicted', 'predicted_variance']:
                result[a] = numpy.array(result[a])

        return result


# ----------------------------------------------------------------------------------------
#  convenience methods
# ----------------------------------------------------------------------------------------

# TODO: authentication

def forecast1D(x, k='SE', n=10, url=None):
    '''
    Forecast the 1D timeseries x for n steps ahead, using kernel k.
    Currently, x must be a numpy array or a python list of floats.
    '''
    DG = TimeSeries1D(x, url=url)
    f = DG.forecast(k, n)
    return f


def auto1D(x, kl=['SE', 'RQ', 'SE + RQ'], n=10, url=None):
    '''
    Train models with kernels in kl on timeseries x.
    Returns a list of models, ordered by prediction accuracy on last n values of x.
    Currently, x must be a numpy array or a python list of floats.
    '''
    DG = TimeSeries1D(x, url=url)
    f = DG.auto(kl, n)
    return f


def summarise(res, top=5):
    '''
    Extracts list of kernel, prediction_error pairs from results of auto1D
    '''
    return [(a['kernel'], a['prediction_error']) for a in res[-top:]]


def version():
    '''
    Return the version number of this API client library.
    '''
    return VERSION
