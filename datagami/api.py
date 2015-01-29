import simplejson as json
import time
import urlparse
import requests
import requests.api

from .exceptions import (
    JobFailedError, NotFound, ValidationError,
    handle_error
)

from .mixins import TimeseriesMixin, RegressionMixin, ClassificationMixin


API_URL = 'https://api.datagami.net'


class Datagami(TimeseriesMixin, RegressionMixin, ClassificationMixin):

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

    def get_model(self, model_key):
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
