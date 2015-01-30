class TimeseriesMixin(object):

    def timeseries_1D_forecast(self, data_key, kernel='SE', steps_ahead=10):
        """ Forecast a one dimensional timeseries """

        data = {
            'data_key': data_key,
            'kernel': kernel,
            'steps_ahead': steps_ahead
        }
        r = self._post('/v1/timeseries/1D/forecast', data)

        model_url = r.json()['url']
        full_response = self.poll(model_url)

        # Return simpler representation
        keys = (
            'fit',
            'fit_variance',
            'predicted',
            'predicted_variance',
            'log_likelihood',
            'parameters',
            'kernel'
        )
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

    def timeseries_ND_train(self, data_key, columns_to_predict, kernel='SE'):

        data = {
            'data_key': data_key,
            'columns_to_predict': columns_to_predict,
            'kernel': kernel
        }
        r = self._post('/v1/timeseries/nD/train', data)

        model_url = r.json()['url']
        response = self.poll(model_url)
        return response

    def timeseries_ND_predict(self, model_key, new_data_key):

        data = {
            'model_key': model_key,
            'new_data_key': new_data_key
        }
        r = self._post('/v1/timeseries/nD/predict', data)

        model_url = r.json()['url']
        response = self.poll(model_url)
        return response
