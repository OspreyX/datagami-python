class RegressionMixin(object):

    def regression_train(
            self, data_key, column_to_predict,
            distribution='gaussian', num_trees=500, depth=1, rate=0.01, cv=2):
        """ Train a regression model """

        data = {
            'data_key': data_key,
            'column_to_predict': column_to_predict,
            'parameters': {
                'distribution': distribution,
                'trees': num_trees,
                'depth': depth,
                'rate': rate,
                'cv': cv
            }
        }
        r = self._post('/v1/regression/train', data)

        model_url = r.json()['url']
        response = self.poll(model_url)
        return response

    def regression_predict(self, model_key, new_data_key):
        """ Make a regression prediction """

        data = {
            'model_key': model_key,
            'new_data_key': new_data_key
        }
        r = self._post('/v1/regression/predict', data)

        model_url = r.json()['url']
        response = self.poll(model_url)
        return response
