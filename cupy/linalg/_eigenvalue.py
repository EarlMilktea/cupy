from __future__ import annotations

import numpy

import cupy
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupy._core import _dtype


def _check_dtype(dtype: numpy.dtype | str) -> None:
    if isinstance(dtype, numpy.dtype):
        dtype = dtype.char
    if dtype not in "fdFD":
        raise RuntimeError(
            "Only float32, float64, complex64, and complex128 are supported"
        )


def _syevd(a, UPLO, with_eigen_vector, overwrite_a=False):
    from cupy_backends.cuda.libs import cublas
    from cupy_backends.cuda.libs import cusolver

    if UPLO not in ('L', 'U'):
        raise ValueError('UPLO argument must be \'L\' or \'U\'')

    # reject_float16=False for backward compatibility
    dtype, v_dtype = _util.linalg_common_type(a, reject_float16=False)
    real_dtype = dtype.char.lower()
    w_dtype = v_dtype.char.lower()

    # Note that cuSolver assumes fortran array
    v = a.astype(dtype, order='F', copy=not overwrite_a)

    m, lda = a.shape
    w = cupy.empty(m, real_dtype)
    dev_info = cupy.empty((), numpy.int32)
    handle = device.Device().cusolver_handle

    if with_eigen_vector:
        jobz = cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = cusolver.CUSOLVER_EIG_MODE_NOVECTOR

    if UPLO == 'L':
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:  # UPLO == 'U'
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    if not runtime.is_hip:
        _check_dtype(dtype)
        type_v = _dtype.to_cuda_dtype(dtype)
        type_w = _dtype.to_cuda_dtype(real_dtype)
        params = cusolver.createParams()
        try:
            work_device_size, work_host_sizse = cusolver.xsyevd_bufferSize(
                handle, params, jobz, uplo, m, type_v, v.data.ptr, lda,
                type_w, w.data.ptr, type_v)
            work_device = cupy.empty(work_device_size, 'b')
            work_host = numpy.empty(work_host_sizse, 'b')
            cusolver.xsyevd(
                handle, params, jobz, uplo, m, type_v, v.data.ptr, lda,
                type_w, w.data.ptr, type_v,
                work_device.data.ptr, work_device_size,
                work_host.ctypes.data, work_host_sizse, dev_info.data.ptr)
        finally:
            cusolver.destroyParams(params)
        cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            cusolver.xsyevd, dev_info)
    else:
        if dtype == 'f':
            buffer_size = cusolver.ssyevd_bufferSize
            syevd = cusolver.ssyevd
        elif dtype == 'd':
            buffer_size = cusolver.dsyevd_bufferSize
            syevd = cusolver.dsyevd
        elif dtype == 'F':
            buffer_size = cusolver.cheevd_bufferSize
            syevd = cusolver.cheevd
        elif dtype == 'D':
            buffer_size = cusolver.zheevd_bufferSize
            syevd = cusolver.zheevd
        else:
            raise RuntimeError('Only float32, float64, complex64, and '
                               'complex128 are supported')

        work_size = buffer_size(
            handle, jobz, uplo, m, v.data.ptr, lda, w.data.ptr)
        work = cupy.empty(work_size, dtype)
        syevd(
            handle, jobz, uplo, m, v.data.ptr, lda,
            w.data.ptr, work.data.ptr, work_size, dev_info.data.ptr)
        cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            syevd, dev_info)

    return w.astype(w_dtype, copy=False), v.astype(v_dtype, copy=False)


def _geev(a, with_eigen_vector, overwrite_a=False):
    from cupy_backends.cuda.libs import cusolver

    if runtime.is_hip:
        raise NotImplementedError("geev is not implemented for HIP")

    dtype, _ = _util.linalg_common_type(a)
    _check_dtype(dtype)
    complex_dtype = numpy.dtype(dtype.char.upper())

    # Force complex-number computation for human-readable output format
    a_ = a.astype(complex_dtype, order='F', copy=not overwrite_a)

    m, lda = a.shape
    w = cupy.empty(m, complex_dtype)
    # Used for both right and (uncomputed) left eigenvectors
    v = cupy.empty_like(a, dtype=complex_dtype, order='F')
    dev_info = cupy.empty((), numpy.int32)
    handle = device.Device().cusolver_handle

    if with_eigen_vector:
        jobvr = cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobvr = cusolver.CUSOLVER_EIG_MODE_NOVECTOR
    # Skip computing left eigenvectors
    jobvl = cusolver.CUSOLVER_EIG_MODE_NOVECTOR

    type_complex = _dtype.to_cuda_dtype(complex_dtype)
    params = cusolver.createParams()
    try:
        work_device_size, work_host_size = cusolver.xgeev_bufferSize(
            handle, params, jobvl, jobvr, m, type_complex, a_.data.ptr, lda,
            type_complex, w.data.ptr, type_complex, v.data.ptr, lda,
            type_complex, v.data.ptr, lda, type_complex)
        work_device = cupy.empty(work_device_size, 'b')
        work_host = numpy.empty(work_host_size, 'b')
        cusolver.xgeev(
            handle, params, jobvl, jobvr, m, type_complex, a_.data.ptr, lda,
            type_complex, w.data.ptr, type_complex, v.data.ptr, lda,
            type_complex, v.data.ptr, lda, type_complex, work_device.data.ptr,
            work_device_size, work_host.ctypes.data, work_host_size,
            dev_info.data.ptr)
    finally:
        cusolver.destroyParams(params)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        cusolver.xgeev, dev_info)

    if all(w.imag == 0.0):
        return w.real, v.real
    return w, v


def eigh(a, UPLO='L'):
    """
    Return the eigenvalues and eigenvectors of a complex Hermitian
    (conjugate symmetric) or a real symmetric matrix.

    Returns two objects, a 1-D array containing the eigenvalues of `a`, and
    a 2-D square array or matrix (depending on the input type) of the
    corresponding eigenvectors (in columns).

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch
            of symmetric 2-D square matrices ``(..., M, M)``.
        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which
            part of ``a`` is used. ``'L'`` uses the lower triangular part of
            ``a``, and ``'U'`` uses the upper triangular part of ``a``.
    Returns:
        tuple of :class:`~cupy.ndarray`:
            Returns a tuple ``(w, v)``. ``w`` contains eigenvalues and
            ``v`` contains eigenvectors. ``v[:, i]`` is an eigenvector
            corresponding to an eigenvalue ``w[i]``. For batch input,
            ``v[k, :, i]`` is an eigenvector corresponding to an eigenvalue
            ``w[k, i]`` of ``a[k]``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.eigh`
    """
    import cupyx.cusolver
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)

    if a.size == 0:
        _, v_dtype = _util.linalg_common_type(a)
        w_dtype = v_dtype.char.lower()
        w = cupy.empty(a.shape[:-1], w_dtype)
        v = cupy.empty(a.shape, v_dtype)
        return w, v

    if a.ndim > 2 or runtime.is_hip:
        w, v = cupyx.cusolver.syevj(a, UPLO, True)
        return w, v
    else:
        return _syevd(a, UPLO, True)


def eig(a):
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)

    if a.size == 0:
        _, v_dtype = _util.linalg_common_type(a)
        w = cupy.empty(a.shape[:-1], v_dtype)
        v = cupy.empty(a.shape, v_dtype)
        return w, v

    if a.ndim == 2:
        return _geev(a, True)

    work = [_geev(a[ind, :, :], True) for ind in numpy.ndindex(a.shape[:-2])]
    w = cupy.stack([x[0] for x in work])
    v = cupy.stack([x[1] for x in work])
    return w.reshape(a.shape[:-1]), v.reshape(a.shape)


def eigvalsh(a, UPLO='L'):
    """
    Compute the eigenvalues of a complex Hermitian or real symmetric matrix.

    Main difference from eigh: the eigenvectors are not computed.

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch
            of symmetric 2-D square matrices ``(..., M, M)``.
        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which
            part of ``a`` is used. ``'L'`` uses the lower triangular part of
            ``a``, and ``'U'`` uses the upper triangular part of ``a``.
    Returns:
        cupy.ndarray:
            Returns eigenvalues as a vector ``w``. For batch input,
            ``w[k]`` is a vector of eigenvalues of matrix ``a[k]``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.eigvalsh`
    """
    import cupyx.cusolver
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)

    if a.size == 0:
        _, v_dtype = _util.linalg_common_type(a)
        w_dtype = v_dtype.char.lower()
        return cupy.empty(a.shape[:-1], w_dtype)

    if a.ndim > 2 or runtime.is_hip:
        return cupyx.cusolver.syevj(a, UPLO, False)
    else:
        return _syevd(a, UPLO, False)[0]


def eigvals(a):
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)

    if a.size == 0:
        _, v_dtype = _util.linalg_common_type(a)
        w = cupy.empty(a.shape[:-1], v_dtype)
        return w

        return cupy.empty(a.shape[:-1], a.dtype)

    if a.ndim == 2:
        return _geev(a, False)[0]

    work = [
        _geev(a[ind, :, :], False)[0] for ind in numpy.ndindex(a.shape[:-2])
    ]
    return cupy.stack(work).reshape(a.shape[:-1])
