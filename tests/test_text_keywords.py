import os
import datagami
import json
from nose.tools import assert_in


KEY = os.environ.get('TEST_API_KEY')
SECRET = os.environ.get('TEST_API_SECRET')

# Data from Stanford Machine Learning course exercise 8
with open('tests/text_keywords_lsi.json', 'r') as f:
    data = json.load(f)


def test_keywords_convenience():

    training_params = {
        'num_topics': 20,
        'exclude_words': ['brownie']
    }

    keywords = datagami.keywords(KEY, SECRET, data, 'lsi', **training_params)

    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)

    # Expect around 54 keywords
    assert 50 < len(keywords) < 70

    # Check top 5 keywords
    top_5 = sorted_keywords[:5]
    top_5_words = zip(*top_5)[0]

    top_10_words = ('yummy', 'love love', 'loved', 'taste', 'love', 'good', 'great', 'best', 'tasted', 'crunchy')

    for word in top_5_words:
        assert_in(word, top_10_words)
