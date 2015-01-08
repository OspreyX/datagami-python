
class TimeseriesMixin(object):

    # Timeseries
    def timeseries_1D_forecast(self, data_key, kernel='SE', steps_ahead=10):

        data = {
            'data_key': data_key,
            'kernel': kernel,
            'steps_ahead': steps_ahead
        }
        r = self._post('/v1/timeseries/1D/forecast', data)

        model_url = r.json()['url']
        full_response = self.poll(model_url)

        # Return simpler representation
        keys = ('fit', 'fit_variance', 'predicted', 'predicted_variance', 'log_likelihood', 'parameters')
        response = {k: full_response[k] for k in keys}

        return response

    def timeseries_1D_auto(self, data_key, kernel_list=None, out_of_sample_size=10):
        """
        Tries to find the best Gaussian Process model for the given dataset,
        by training a collection of models on a reduced data set
        and ranking them by the accuracy of their prediction.

        Returns a dictionary with a kernel per key
        """

        data = {
            'data_key': data_key,
            'kernel_list': kernel_list,
            'oos_window': out_of_sample_size
        }
        r = self._post('/v1/timeseries/1D/auto', data)

        model_url = r.json()['url']
        full_response = self.poll(model_url)

        return full_response

        # # Return simpler representation
        # models = response['models']
        # response = {}

        # return response


# ----------------------------------------------------------------------------------------


# class TimeSeriesND(Datagami):

#     '''
#     Class to handle all ND timeseries models. Instances contain details about the API connection
#     and references to the uploaded data.  Methods on this object are: get_data, getModel, train, and predict.

#     '''

#     def __init__(self, x, username, token, url=None):
#         '''
#         Create a TimeSeriesND object from input data x.  Note, x must be a
#         dictionary of numpy or list of floats, ie keys are names and values are column vectors.
#         '''

#         super(TimeSeriesND, self).__init__(self, username, token, url)

#         # sanity check on data array, converts numpy to python list
#         y = self.validateArrayND(x)

#         # upload timeseries data to the API
#         data_json = json.dumps(y)
#         r = requests.post(self.data_url, data={'data': data_json}, auth=self.auth)
#         r.raise_for_status()
#         r_data = r.json()

#         if 'data_key' not in r_data:
#             raise requests.exceptions.RequestException(r_data)

#         self.data_key = r_data['data_key']

#     def train(self, columns_to_predict, kernel='SE'):
#         '''
#         Fits a Gaussian Process model with the supplied kernel to input array x.
#         The variables to predict are given by name in the argumet columns_to_predict.
#         Returns a model_keydictionary of results: fit, fit_variance, predicted, predicted_variance.
#         See API documentation for details.
#         '''
#         # validation
#         if type(kernel) not in (str, unicode):
#             raise ValueError("kernel must be text (string or unicode).  Found %s" % type(kernel))

#         if type(columns_to_predict) in (str, unicode):
#             columns_to_predict = [columns_to_predict]
#         elif type(columns_to_predict) not in (tuple, list):
#             raise ValueError("columns_to_predict must be a single text or tuple/list of text")
#         elif type(columns_to_predict) in (tuple, list):
#             if not all([type(s) in (str, unicode) for s in columns_to_predict]):
#                 raise ValueError("columns_to_predict must contain only text or unicode")
#         columns_to_predict = json.dumps(columns_to_predict)

#         # post to train endpoint
#         params_dict = {'data_key': self.data_key, 'kernel': kernel, 'columns_to_predict': columns_to_predict}
#         r = requests.post(self.ts_trainND_url, data=params_dict, auth=self.auth)
#         r.raise_for_status()

#         # get results from server, synchronous, ie poll and wait
#         result = self.poll_retrieve(r.json())

#         # store some values
#         self.model_key = result['model_key']

#         # clean up object for return to user
#         # result.pop('job_id', None)
#         result.pop('status', None)
#         result.pop('data_key', None)
#         result.pop('model_key', None)
#         result.pop('type', None)
#         result.pop('steps_ahead', None)

#         # return numpy if input was numpy
#         if self.data_type is numpy.array:
#             for a in ['fit', 'fit_variance', 'predicted', 'predicted_variance']:
#                 result[a] = numpy.array(result[a])

#         return result

#     def predict(self, newdata):
#         '''
#         Fits a Gaussian Process model with the supplied kernel to input array x.
#         The variables to predict are given by name in the argumet columns_to_predict.
#         Returns a model_keydictionary of results: fit, fit_var, pred, pred_var.
#         See API documentation for details.
#         '''
#         # validation
#         y = self.validateArrayND(newdata)

#         # upoload newdata
#         newdata_json = json.dumps(y)
#         r = requests.post(self.data_url, data={'data': newdata_json}, auth=self.auth)
#         r.raise_for_status()
#         r_data = r.json()
#         if 'data_key' not in r_data:
#             raise requests.exceptions.RequestException(r_data)
#         self.newdata_key = r_data['data_key']

#         # post to train endpoint
#         params_dict = {'new_data_key': self.newdata_key, 'model_key': self.model_key}
#         r = requests.post(self.ts_predictND_url, data=params_dict, auth=self.auth)
#         r.raise_for_status()

#         # get results from server, synchronous, ie poll and wait
#         result = self.poll_retrieve(r.json())

#         # store some values
#         self.model_key = result['model_key']

#         # clean up object for return to user
#         # result.pop('job_id', None)
#         result.pop('status', None)
#         result.pop('data_key', None)
#         result.pop('new_data_key', None)
#         result.pop('model_key', None)
#         result.pop('type', None)
#         result.pop('message', None)

#         # return numpy if input was numpy
#         if self.data_type is numpy.array:
#             for a in ['fit', 'fit_variance', 'predicted', 'predicted_variance']:
#                 result[a] = numpy.array(result[a])

#         return result

