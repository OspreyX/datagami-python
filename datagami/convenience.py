from .api import Datagami

""" Convenience methods """


def forecast1D(data, key, secret, kernel='SE', steps_ahead=10):
    """
    Forecast the 1D timeseries x for n steps ahead, using kernel k.
    Currently, x must be a numpy array or a python list of floats.
    """
    d = Datagami(key, secret)
    data_key = d.upload_data(data)
    forecast = d.timeseries_1D_forecast(data_key, kernel, steps_ahead)
    return forecast


def auto1D(data, key, secret, kernel_list=None, out_of_sample_size=10):
    """
    Train models with kernels in kl on timeseries x.
    Returns a list of models, ordered by prediction accuracy on last n values of x.
    Currently, data must be a python list of floats.
    """
    if kernel_list is None:
        kernel_list = ['SE', 'RQ', 'SE + RQ']

    d = Datagami(key, secret)
    data_key = d.upload_data(data)
    forecast = d.timeseries_1D_forecast(data_key, kernel_list, out_of_sample_size)
    return forecast


# def summarise(res, top=5):
#     """
#     Extracts list of kernel, prediction_error pairs from results of auto1D
#     """
#     return [(a['kernel'], a['prediction_error']) for a in res[-top:]]
