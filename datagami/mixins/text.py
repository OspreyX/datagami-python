class TextMixin(object):

    def text_keywords(self, data_key, method='lsi', num_topics=20, exclude_words=None):
        """ Extract keywords from text """

        if exclude_words is None:
            exclude_words = []

        data = {
            'data_key': data_key,
            'method': method,
            'num_topics': num_topics,
            'exclude_words': exclude_words
        }
        r = self._post('/v1/text/keywords', data)

        model_url = r.json()['url']
        response = self.poll(model_url)

        return response
