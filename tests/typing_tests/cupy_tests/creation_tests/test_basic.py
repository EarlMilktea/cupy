from __future__ import annotations

import numpy
import pytest

import cupy


def test_empty() -> None:
    cupy.empty(10)
    cupy.empty((10, 20))
    cupy.empty(numpy.array([10, 20]))
    cupy.empty((10, 20), float)
    cupy.empty((10, 20), int)
    cupy.empty((10, 20), numpy.float32)
    cupy.empty((10, 20), "i4")
    cupy.empty((10, 20), "int32")
    cupy.empty((10, 20), float, "C")
    cupy.empty((10, 20), float, "F")
    cupy.empty(shape=(10, 20), dtype=float, order="C")


@pytest.mark.xfail
def test_empty_ng() -> None:
    cupy.empty()  # type: ignore[call-arg]
    cupy.empty(10, 20)  # type: ignore[arg-type]
    cupy.empty((10, 20), float, "K")  # type: ignore[arg-type]
    cupy.empty((10, 20), float, "A")  # type: ignore[arg-type]
    cupy.empty((10, 20), float, "X")  # type: ignore[arg-type]
    # TODO(asi1024): Fix to fail typecheck
    # cupy.empty(cupy.array([10, 20]))
    # cupy.empty((10, 20), numpy.datetime64)


def test_empty_like() -> None:
    x = cupy.empty((10, 20))
    cupy.empty_like(x)
    cupy.empty_like(x, float)
    cupy.empty_like(x, float, "C")
    cupy.empty_like(x, float, "F")
    cupy.empty_like(x, float, "K")
    cupy.empty_like(x, float, "A")
    cupy.empty_like(prototype=x, dtype=float, order="C", shape=(10, 20))


def test_eye() -> None:
    cupy.eye(10)
    cupy.eye(10, 20)
    cupy.eye(10, 20, 3)
    cupy.eye(10, 20, 3, float, "C")
    cupy.eye(10, 20, 3, float, "F")
    cupy.eye(N=10, M=20, k=3, dtype=float, order="C")


@pytest.mark.xfail
def test_eye_ng() -> None:
    cupy.eye(10, 20, 3, float, "K")  # type: ignore[arg-type]
    cupy.eye(10, 20, 3, float, "A")  # type: ignore[arg-type]


def test_identity() -> None:
    cupy.identity(10)
    cupy.identity(10, float)
    cupy.identity(n=10, dtype=float)


def test_ones() -> None:
    cupy.ones((10, 20))
    cupy.ones((10, 20), float)
    cupy.ones((10, 20), float, "C")
    cupy.ones((10, 20), float, "F")


@pytest.mark.xfail
def test_ones_ng() -> None:
    cupy.ones((10, 20), float, "K")  # type: ignore[arg-type]
    cupy.ones((10, 20), float, "A")  # type: ignore[arg-type]


def test_ones_like() -> None:
    x = cupy.ones((10, 20))
    cupy.ones_like(x)
    cupy.ones_like(x, float)
    cupy.ones_like(x, float, "C")
    cupy.ones_like(x, float, "F")
    cupy.ones_like(x, float, "K")
    cupy.ones_like(x, float, "A")
    cupy.ones_like(a=x, dtype=float, order="C", shape=(10, 20))


def test_zeros() -> None:
    cupy.zeros((10, 20))
    cupy.zeros((10, 20), float)
    cupy.zeros((10, 20), float, "C")
    cupy.zeros((10, 20), float, "F")


@pytest.mark.xfail
def test_zeros_ng() -> None:
    cupy.zeros((10, 20), float, "K")  # type: ignore[arg-type]
    cupy.zeros((10, 20), float, "A")  # type: ignore[arg-type]


def test_zeros_like() -> None:
    x = cupy.zeros((10, 20))
    cupy.zeros_like(x)
    cupy.zeros_like(x, float)
    cupy.zeros_like(x, float, "C")
    cupy.zeros_like(x, float, "F")
    cupy.zeros_like(x, float, "K")
    cupy.zeros_like(x, float, "A")
    cupy.zeros_like(a=x, dtype=float, order="C", shape=(10, 20))


def test_full() -> None:
    cupy.full((10, 20), 30)
    cupy.full((10, 20), 30, float)
    cupy.full((10, 20), 30, float, "C")
    cupy.full((10, 20), 30, float, "F")
    # cupy.full((10, 20), "abc")  # TODO(asi1024): Fix to fail typecheck


@pytest.mark.xfail
def test_full_ng() -> None:
    cupy.full((10, 20))  # type: ignore[call-arg]
    cupy.full((10, 20), 30, float, "K")  # type: ignore[arg-type]
    cupy.full((10, 20), 30, float, "A")  # type: ignore[arg-type]


def test_full_like() -> None:
    x = cupy.full((10, 20), 30)
    cupy.full_like(x, 30)
    cupy.full_like(x, 30, float)
    cupy.full_like(x, 30, float, "C")
    cupy.full_like(x, 30, float, "F")
    cupy.full_like(x, 30, float, "K")
    cupy.full_like(x, 30, float, "A")
    cupy.full_like(a=x, fill_value=30, dtype=float, order="C", shape=(10, 20))
