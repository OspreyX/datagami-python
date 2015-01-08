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


class ConnectionError(DatagamiException, requests.exceptions.ConnectionError):
    """
    Connection to the server has failed for some reason - for example,
    if we can't connect to the internet
    """
    pass


class ValidationError(DatagamiException, ValueError):
    """ Something was wrong with your request """
    pass

    def __str__(self):
        msg = self.args[0]

        if self.response:
            error_message = self.response.get('message')

            if error_message:
                msg = '{}: {}'.format(msg, error_message)

        return msg


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
