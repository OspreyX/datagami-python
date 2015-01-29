from .api import Datagami
from .convenience import (
    forecast1D,
    auto1D,
    regression
)

__title__ = 'datagami'
__version__ = '1.0.3'
__license__ = 'MIT'
__copyright__ = 'Copyright 2015 Datagami'


def version():
    return __version__
