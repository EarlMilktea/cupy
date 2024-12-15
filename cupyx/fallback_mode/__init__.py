import warnings
from cupy import _util

# Attributes and Methods for fallback_mode
# Auto-execute numpy method when corresponding cupy method is not found

# "NOQA" to suppress flake8 warning
from cupyx.fallback_mode.fallback import numpy  # NOQA


_util.experimental('cupyx.fallback_mode.numpy')

warnings.warn(
    '`cupyx.fallback` is deprecated and will be removed in a future version '
    'of CuPy.',
    DeprecationWarning
)
