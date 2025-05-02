from __future__ import annotations

import functools
import glob
import os
import sys
from collections.abc import Callable
from typing import Any


def _memoize(f: Callable) -> Callable:
    memo = {}

    @functools.wraps(f)
    def ret(*args: Any, **kwargs: Any) -> Any:
        key = (args, frozenset(kwargs.items()))
        if key not in memo:
            memo[key] = f(*args, **kwargs)
        return memo[key]
    return ret


@_memoize
def get_nvtx_path() -> str | None:
    from cupy_builder import logger

    assert sys.platform == 'win32'

    prog = os.environ.get('ProgramFiles', 'C:\\Program Files')
    pattern = os.path.join(
        prog, 'NVIDIA Corporation', 'Nsight Systems *', 'target-windows-x64',
        'nvtx',
    )
    logger.info('Looking for NVTX: %s', pattern)
    candidates = sorted(glob.glob(pattern))
    if len(candidates) != 0:
        # Pick the latest one
        nvtx = candidates[-1]
        logger.info('Using NVTX at: %s', nvtx)
        return nvtx
    if os.environ.get('CONDA_BUILD', '0') == '1':
        return os.environ['PREFIX']
    logger.warning('NVTX could not be found')
    return None
