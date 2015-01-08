import json
import logging
import time
import urlparse
import requests
import requests.api

from datagami.mixins import TimeseriesMixin
from datagami.exceptions import (
    JobFailedError, NotFound, ValidationError,
    handle_error
)


VERSION = '1.0.3'

logger = logging.getLogger(__name__)
log_format = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=log_format)
logger.setLevel(logging.INFO)


# API_URL = 'http://beta.api.datagami.net'
API_URL = 'https://develop-api.datagami.net'


class Datagami(TimeseriesMixin):

    def __init__(self, username, token, url=None):

        self._auth = (username, token)

        if url is None:
            url = API_URL

        self.base_url = url

    def _request(self, method, path, **kwargs):
        """ Make an authenticated request to the Datagami API """

        url = urlparse.urljoin(self.base_url, path)

        with handle_error():
            r = requests.api.request(method, url, auth=self._auth, **kwargs)
            r.raise_for_status()

        return r

    def _get(self, path):
        """ Send a GET request to the Datagami API at BASE_URL/`path` """
        return self._request('get', path)

    def _post(self, path, data):
        """ Send POST `data` to the Datagami API at BASE_URL/`path` """
        return self._request('post', path, data=json.dumps(data))

    def _delete(self, path):
        """ Send a DELETE request to the Datagami API at BASE_URL/`path` """
        return self._request('delete', path)

    def upload_data(self, data):
        """ Upload data to API, return data_key """

        r = self._post('/v1/data', {'data': data})
        return r.json()['data_key']

    def get_data(self, data_key):
        """ Fetch data from API """

        data_url = '/v1/data/{}'.format(data_key)
        r = self._get(data_url)
        return r.json()['data']

    def delete_data(self, data_key):
        """ Delete data from API """

        data_url = '/v1/data/{}'.format(data_key)
        try:
            r = self._delete(data_url)
            return r.json()['status'] == 'SUCCESS'
        except NotFound:
            return False

    def get_model(self, model_key, poll=True):
        """ Poll model, fetch from API when status is SUCCESS """

        model_url = '/v1/model/{}'.format(model_key)
        return self.poll(model_url)

    def poll(self, url, max_tries=1000):
        """ Poll given url until returned status is SUCCESS, then return response """

        delay = 1
        for counter in range(max_tries):

            r = self._get(url)
            response = r.json()

            if response['status'] == 'SUCCESS':
                return response

            elif response['status'] == 'FAILURE':
                raise JobFailedError("Job failed", response=response)

            time.sleep(delay)

            # Try 5x 1 second delays, then 2s, 3s, etc
            if counter > 5:
                delay += 1

        raise JobFailedError("Job timeout")


# ----------------------------------------------------------------------------------------
#  convenience methods
# ----------------------------------------------------------------------------------------

def forecast1D(data, key, secret, kernel='SE', steps_ahead=10):
    '''
    Forecast the 1D timeseries x for n steps ahead, using kernel k.
    Currently, x must be a numpy array or a python list of floats.
    '''
    d = Datagami(key, secret)
    data_key = d.upload_data(data)
    forecast = d.timeseries_1D_forecast(data_key, kernel, steps_ahead)
    return forecast


def auto1D(data, key, secret, kernel_list=None, out_of_sample_size=10):
    '''
    Train models with kernels in kl on timeseries x.
    Returns a list of models, ordered by prediction accuracy on last n values of x.
    Currently, data must be a python list of floats.
    '''
    if kernel_list is None:
        kernel_list = ['SE', 'RQ', 'SE + RQ']

    d = Datagami(key, secret)
    data_key = d.upload_data(data)
    forecast = d.timeseries_1D_forecast(data_key, kernel_list, out_of_sample_size)
    return forecast


# def summarise(res, top=5):
#     '''
#     Extracts list of kernel, prediction_error pairs from results of auto1D
#     '''
#     return [(a['kernel'], a['prediction_error']) for a in res[-top:]]


def version():
    '''
    Return the version number of this API client library.
    '''
    return VERSION
