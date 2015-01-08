try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='datagami-python',
    version='0.0.1',
    author='Datagami',
    author_email='hello@datagami.info',
    packages=['datagami'],
    url='https://github.com/datagami/datagami-python',
    description='Datagami library for Python',
    long_description=open('README.md').read(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2 :: Only',
    ]
)
