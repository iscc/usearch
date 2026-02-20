#!/usr/bin/env -S uv run --quiet --script
"""
USearch Distance Functions Tests

Test suite for various distance metrics and their implementations in USearch,
including cosine, L2, inner product, and custom distance functions.

Usage:
    uv run python/scripts/test_distances.py
    
Dependencies listed in the script header for uv to resolve automatically.
"""
# /// script
# dependencies = [
#   "pytest",
#   "numpy",
#   "usearch"
# ]
# ///

import pytest
import numpy as np

import usearch
from usearch.eval import random_vectors
from usearch.index import search

from usearch.index import (
    Index,
    MetricKind,
    ScalarKind,
)


@pytest.mark.parametrize(
    "metric",
    [
        MetricKind.Cos,
        MetricKind.L2sq,
        MetricKind.Divergence,
        MetricKind.Pearson,
    ],
)
@pytest.mark.parametrize(
    "quantization",
    [
        ScalarKind.F32,
        ScalarKind.F16,
        ScalarKind.BF16,
        ScalarKind.I8,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.float16,
        np.int8,
    ],
)
def test_distances_continuous(metric, quantization, dtype):
    ndim = 1024
    try:
        index = Index(ndim=ndim, metric=metric, dtype=quantization)
        vectors = random_vectors(count=2, ndim=ndim, dtype=dtype)
        keys = np.arange(2)
        index.add(keys, vectors)
    except ValueError:
        pytest.skip(f"Unsupported metric `{metric}`, quantization `{quantization}`, dtype `{dtype}`")
        return

    rtol = 1e-2
    atol = 1e-2

    distance_itself_first = index.pairwise_distance([0], [0])
    distance_itself_second = index.pairwise_distance([1], [1])
    distance_different = index.pairwise_distance([0], [1])

    assert not np.allclose(distance_different, 0)
    assert np.allclose(distance_itself_first, 0, rtol=rtol, atol=atol)
    assert np.allclose(distance_itself_second, 0, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "metric",
    [
        MetricKind.Hamming,
        MetricKind.Tanimoto,
        MetricKind.Sorensen,
    ],
)
def test_distances_sparse(metric):
    ndim = 1024
    index = Index(ndim=ndim, metric=metric, dtype=ScalarKind.B1)
    vectors = random_vectors(count=2, ndim=ndim, dtype=ScalarKind.B1)
    keys = np.arange(2)
    index.add(keys, vectors)

    rtol = 1e-2
    atol = 1e-2

    distance_itself_first = index.pairwise_distance([0], [0])
    distance_itself_second = index.pairwise_distance([1], [1])
    distance_different = index.pairwise_distance([0], [1])

    assert not np.allclose(distance_different, 0)
    assert np.allclose(distance_itself_first, 0, rtol=rtol, atol=atol) and np.allclose(
        distance_itself_second, 0, rtol=rtol, atol=atol
    )


def _make_nphd_vector(length_byte, data_bytes, ndim_bytes=33):
    """Build a length-prefixed binary vector padded to ndim_bytes."""
    vec = np.zeros(ndim_bytes, dtype=np.uint8)
    vec[0] = length_byte
    for i, b in enumerate(data_bytes):
        vec[1 + i] = b
    return vec


def test_nphd_identical_vectors():
    """Identical vectors should have distance 0.0."""
    ndim = 264  # 33 bytes * 8 bits
    index = Index(ndim=ndim, metric=MetricKind.NPHD, dtype=ScalarKind.B1)
    vec = _make_nphd_vector(4, [0xAA, 0xBB, 0xCC, 0xDD])
    index.add(0, vec)
    index.add(1, vec.copy())
    dist = index.pairwise_distance([0], [1])
    assert dist[0] == pytest.approx(0.0), f"Expected 0.0, got {dist[0]}"


def test_nphd_maximally_different():
    """All-zeros vs all-ones prefix should give distance 1.0."""
    ndim = 264
    index = Index(ndim=ndim, metric=MetricKind.NPHD, dtype=ScalarKind.B1)
    a = _make_nphd_vector(4, [0x00, 0x00, 0x00, 0x00])
    b = _make_nphd_vector(4, [0xFF, 0xFF, 0xFF, 0xFF])
    index.add(0, a)
    index.add(1, b)
    dist = index.pairwise_distance([0], [1])
    assert dist[0] == pytest.approx(1.0), f"Expected 1.0, got {dist[0]}"


def test_nphd_known_values():
    """Verify NPHD against hand-calculated values."""
    ndim = 264
    index = Index(ndim=ndim, metric=MetricKind.NPHD, dtype=ScalarKind.B1)
    # a: length=2, data=[0b10101010, 0b11001100]
    # b: length=2, data=[0b10101010, 0b11001111]
    # XOR byte 0: 0x00 -> 0 bits differ
    # XOR byte 1: 0xCC ^ 0xCF = 0x03 -> 2 bits differ
    # hamming = 2, min_bits = 16, nphd = 2/16 = 0.125
    a = _make_nphd_vector(2, [0xAA, 0xCC])
    b = _make_nphd_vector(2, [0xAA, 0xCF])
    index.add(0, a)
    index.add(1, b)
    dist = index.pairwise_distance([0], [1])
    assert dist[0] == pytest.approx(0.125), f"Expected 0.125, got {dist[0]}"


def test_nphd_zero_length_prefix():
    """Length byte 0 should return distance 0.0."""
    ndim = 264
    index = Index(ndim=ndim, metric=MetricKind.NPHD, dtype=ScalarKind.B1)
    a = _make_nphd_vector(0, [])
    b = _make_nphd_vector(4, [0xFF, 0xFF, 0xFF, 0xFF])
    index.add(0, a)
    index.add(1, b)
    dist = index.pairwise_distance([0], [1])
    assert dist[0] == pytest.approx(0.0), f"Expected 0.0, got {dist[0]}"


def test_nphd_prefix_compatibility():
    """Shorter prefix of longer vector should compare only common prefix."""
    ndim = 264
    index = Index(ndim=ndim, metric=MetricKind.NPHD, dtype=ScalarKind.B1)
    # a: length=2, data=[0xFF, 0xFF, ...]
    # b: length=4, data=[0xFF, 0xFF, 0x00, 0x00]
    # Common prefix = min(2, 4) = 2 bytes
    # XOR over 2 bytes: all zeros -> hamming = 0
    # nphd = 0 / 16 = 0.0
    a = _make_nphd_vector(2, [0xFF, 0xFF])
    b = _make_nphd_vector(4, [0xFF, 0xFF, 0x00, 0x00])
    index.add(0, a)
    index.add(1, b)
    dist = index.pairwise_distance([0], [1])
    assert dist[0] == pytest.approx(0.0), f"Expected 0.0, got {dist[0]}"


def test_nphd_length_byte_excluded():
    """The length byte itself must not be included in Hamming computation."""
    ndim = 264
    index = Index(ndim=ndim, metric=MetricKind.NPHD, dtype=ScalarKind.B1)
    # Both have length=1, but different length byte values wouldn't matter
    # since length byte is not compared. Same data byte -> distance 0.
    a = _make_nphd_vector(1, [0xAA])
    b = _make_nphd_vector(1, [0xAA])
    # Corrupt length bytes to be different but still valid
    # (both claim length=1, data is identical)
    index.add(0, a)
    index.add(1, b)
    dist = index.pairwise_distance([0], [1])
    assert dist[0] == pytest.approx(0.0), f"Expected 0.0, got {dist[0]}"

    # Now verify with different data: a=0xAA, b=0x55
    # 0xAA ^ 0x55 = 0xFF -> 8 bits differ, min_bits=8, nphd=1.0
    c = _make_nphd_vector(1, [0x55])
    index.add(2, c)
    dist2 = index.pairwise_distance([0], [2])
    assert dist2[0] == pytest.approx(1.0), f"Expected 1.0, got {dist2[0]}"


def test_nphd_index_roundtrip(tmp_path):
    """Save/load index and verify NPHD metric restores correctly."""
    ndim = 264
    index = Index(ndim=ndim, metric=MetricKind.NPHD, dtype=ScalarKind.B1)
    a = _make_nphd_vector(4, [0xAA, 0xBB, 0xCC, 0xDD])
    b = _make_nphd_vector(4, [0xAA, 0xBB, 0xCC, 0x00])
    index.add(0, a)
    index.add(1, b)

    dist_before = index.pairwise_distance([0], [1])

    path = str(tmp_path / "nphd_index.usearch")
    index.save(path)

    loaded = Index.restore(path)
    assert str(loaded.metric_kind) == "MetricKind.NPHD"
    dist_after = loaded.pairwise_distance([0], [1])
    assert dist_after[0] == pytest.approx(dist_before[0])


def test_nphd_search():
    """Verify nearest neighbor search returns correct results."""
    ndim = 264
    index = Index(ndim=ndim, metric=MetricKind.NPHD, dtype=ScalarKind.B1)

    # Add 3 vectors with length=4
    target = _make_nphd_vector(4, [0xAA, 0xBB, 0xCC, 0xDD])
    close = _make_nphd_vector(4, [0xAA, 0xBB, 0xCC, 0xDE])  # 2 bits differ
    far = _make_nphd_vector(4, [0x55, 0x44, 0x33, 0x22])  # many bits differ

    index.add(0, target)
    index.add(1, close)
    index.add(2, far)

    results = index.search(target, 3)
    keys = results.keys.flatten().tolist()
    distances = results.distances.flatten().tolist()

    # Target should be nearest (distance 0), then close, then far
    assert keys[0] == 0
    assert distances[0] == pytest.approx(0.0)
    assert keys[1] == 1
    assert distances[1] < distances[2]
