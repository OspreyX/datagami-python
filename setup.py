import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='datagami-python',
    version='0.0.1',
    author='Datagami',
    author_email='hello@datagami.info',
    packages=find_packages(),
    install_requires=[
        'requests',
        'simplejson'
    ],
    url='https://github.com/datagami/datagami-python',
    description='Datagami library for Python',
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2 :: Only',
    ]
)
