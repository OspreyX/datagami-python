""" datagami.exceptions

This module contains the set of Datagami's exceptions.
"""

import requests.exceptions
from contextlib import contextmanager


class DatagamiException(IOError):
    """
    There was an ambiguous exception that occurred while handling your
    request
    """

    def __init__(self, *args, **kwargs):
        self.response = kwargs.pop('response', None)

        super(DatagamiException, self).__init__(*args, **kwargs)

    def __str__(self):
        msg = self.args[0]

        if self.response:
            message = self.response.get('message', '')
            error = self.response.get('error', '')

            msg = ': '.join([msg, message, error])

        return msg


class ConnectionError(DatagamiException, requests.exceptions.ConnectionError):
    """
    Connection to the server has failed for some reason - for example,
    if we can't connect to the internet
    """
    pass


class ValidationError(DatagamiException, ValueError):
    """ Something was wrong with your request """
    pass


class JobFailedError(DatagamiException, ValueError):
    """ The machine learning job has failed """
    pass


class NotFound(DatagamiException):
    """ This resource can't be found """
    pass


@contextmanager
def handle_error():
    try:
        yield

    except requests.exceptions.Timeout:
        msg = 'Connection timeout'
        raise ConnectionError(msg)

    except requests.exceptions.ConnectionError:
        msg = 'Connection failure'
        raise ConnectionError(msg)

    except requests.exceptions.HTTPError as e:

        code = e.response.status_code

        if code == 400:
            raise ValidationError('Invalid request', response=e.response.json())

        elif code == 404:
            raise NotFound('Not Found', response=e.response.json())

        elif code == 500:
            raise DatagamiException('Server error', response=e.response.json())

        else:
            raise
