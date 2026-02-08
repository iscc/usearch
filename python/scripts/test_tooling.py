#!/usr/bin/env -S uv run --quiet --script
"""
USearch Tooling Tests

Test suite for USearch utility functions and tooling,
including I/O operations, matrix handling, and helper functions.

Usage:
    uv run python/scripts/test_tooling.py
    
Dependencies listed in the script header for uv to resolve automatically.
"""
# /// script
# dependencies = [
#   "pytest",
#   "numpy",
#   "usearch"
# ]
# ///

import os

import pytest
import numpy as np

import usearch
from usearch.io import load_matrix, save_matrix
from usearch.index import search
from usearch.eval import random_vectors

from usearch.index import Match, Matches, BatchMatches, Index, Indexes, kmeans


dimensions = [3, 97, 256]
batch_sizes = [1, 77, 100]


@pytest.mark.parametrize("rows", batch_sizes)
@pytest.mark.parametrize("cols", dimensions)
def test_serializing_fbin_matrix(rows: int, cols: int):
    """
    Test the serialization of floating point binary matrix.

    :param int rows: The number of rows in the matrix.
    :param int cols: The number of columns in the matrix.
    """
    original = np.random.rand(rows, cols).astype(np.float32)
    save_matrix(original, "tmp.fbin")
    reconstructed = load_matrix("tmp.fbin")
    assert np.allclose(original, reconstructed)
    os.remove("tmp.fbin")


@pytest.mark.parametrize("rows", batch_sizes)
@pytest.mark.parametrize("cols", dimensions)
def test_serializing_ibin_matrix(rows: int, cols: int):
    """
    Test the serialization of integer binary matrix.

    :param int rows: The number of rows in the matrix.
    :param int cols: The number of columns in the matrix.
    """
    original = np.random.randint(0, rows + 1, size=(rows, cols)).astype(np.int32)
    save_matrix(original, "tmp.ibin")
    reconstructed = load_matrix("tmp.ibin")
    assert np.allclose(original, reconstructed)
    os.remove("tmp.ibin")


@pytest.mark.parametrize("rows", batch_sizes)
@pytest.mark.parametrize("cols", dimensions)
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("reordered", [False, True])
def test_exact_search(rows: int, cols: int, k: int, reordered: bool):
    """
    Test exact search.

    :param int rows: The number of rows in the matrix.
    :param int cols: The number of columns in the matrix.
    """
    if cols < 10:
        pytest.skip("In low dimensions collisions are likely.")
    original = np.random.rand(rows, cols)
    keys = np.arange(rows)
    k = min(k, rows)

    if reordered:
        reordered_keys = np.arange(rows)
        np.random.shuffle(reordered_keys)
    else:
        reordered_keys = keys

    matches: BatchMatches = search(original, original[reordered_keys], k, exact=True)
    top_matches = [int(m.keys[0]) for m in matches] if rows > 1 else [int(matches.keys[0])]
    assert top_matches == list(reordered_keys)

    matches: Matches = search(original, original[-1], k, exact=True)
    top_match = int(matches.keys[0])
    assert top_match == keys[-1]


def test_matches_creation_and_methods():
    matches = Matches(
        keys=np.array([1, 2]),
        distances=np.array([0.5, 0.6]),
        visited_members=2,
        computed_distances=2,
    )
    assert len(matches) == 2
    assert matches[0] == Match(key=1, distance=0.5)
    assert matches.to_list() == [(1, 0.5), (2, 0.6)]


def test_batch_matches_creation_and_methods():
    keys = np.array([[1, 2], [3, 4]])
    distances = np.array([[0.5, 0.6], [0.7, 0.8]])
    counts = np.array([2, 2])
    batch_matches = BatchMatches(
        keys=keys,
        distances=distances,
        counts=counts,
        visited_members=2,
        computed_distances=2,
    )

    assert len(batch_matches) == 2
    assert batch_matches[0].keys.tolist() == [1, 2]
    assert batch_matches[0].distances.tolist() == [0.5, 0.6]
    assert batch_matches.to_list() == [(1, 0.5), (2, 0.6), (3, 0.7), (4, 0.8)]


def test_multi_index():
    ndim = 10
    index_a = Index(ndim=ndim)
    index_b = Index(ndim=ndim)

    vectors = random_vectors(count=3, ndim=ndim)
    index_a.add(42, vectors[0])
    index_b.add(43, vectors[1])

    indexes = Indexes([index_a, index_b])

    # Search top 10 for 1
    matches = indexes.search(vectors[2], 10)
    assert len(matches) == 2

    # Search top 1 for 1
    matches = indexes.search(vectors[2], 1)
    assert len(matches) == 1

    # Search top 10 for 3
    matches = indexes.search(vectors, 10)
    assert len(matches) == 3
    assert len(matches[0].keys) == 2


def test_multi_index_uuid():
    """Indexes with UUID (128-bit) keyed shards."""
    ndim = 10
    index_a = Index(ndim=ndim, key_kind="uuid")
    index_b = Index(ndim=ndim, key_kind="uuid")

    vectors = random_vectors(count=3, ndim=ndim)
    key_a = b"\x00" * 15 + b"\x01"
    key_b = b"\x00" * 15 + b"\x02"
    index_a.add(key_a, vectors[0])
    index_b.add(key_b, vectors[1])

    indexes = Indexes([index_a, index_b])
    assert len(indexes) == 2

    matches = indexes.search(vectors[2], 10)
    assert len(matches) == 2

    matches = indexes.search(vectors[2], 1)
    assert len(matches) == 1

    matches = indexes.search(vectors, 10)
    assert len(matches) == 3
    assert len(matches[0].keys) == 2


def test_multi_index_uuid_paths(tmp_path):
    """Indexes with UUID shards loaded from file paths."""
    ndim = 10
    index_a = Index(ndim=ndim, key_kind="uuid")
    index_b = Index(ndim=ndim, key_kind="uuid")

    vectors = random_vectors(count=3, ndim=ndim)
    index_a.add(b"\x00" * 15 + b"\x01", vectors[0])
    index_b.add(b"\x00" * 15 + b"\x02", vectors[1])

    path_a = str(tmp_path / "a.usearch")
    path_b = str(tmp_path / "b.usearch")
    index_a.save(path_a)
    index_b.save(path_b)

    # Path-only init with auto key-kind detection
    # Use threads=1 to avoid pre-existing multi-threaded merge_paths crash on Windows
    indexes = Indexes(paths=[path_a, path_b], view=True, threads=1)
    assert len(indexes) == 2
    matches = indexes.search(vectors[2], 10)
    assert len(matches) == 2

    # merge_path method
    indexes2 = Indexes(key_kind="uuid")
    indexes2.merge_path(path_a)
    assert len(indexes2) == 1


def test_multi_index_merge_path(tmp_path):
    """merge_path works for u64 indexes."""
    ndim = 10
    index = Index(ndim=ndim)
    vectors = random_vectors(count=1, ndim=ndim)
    index.add(42, vectors[0])

    path = str(tmp_path / "shard.usearch")
    index.save(path)

    indexes = Indexes()
    indexes.merge_path(path)
    assert len(indexes) == 1

    matches = indexes.search(vectors[0], 1)
    assert len(matches) == 1


def test_multi_index_mixed_key_kinds_rejected():
    """Merging indexes with different key kinds raises TypeError."""
    ndim = 10
    index_u64 = Index(ndim=ndim, key_kind="u64")
    index_uuid = Index(ndim=ndim, key_kind="uuid")

    indexes = Indexes([index_u64])
    with pytest.raises(TypeError, match="key_kind"):
        indexes.merge(index_uuid)


def test_multi_index_empty():
    """Empty Indexes behaves correctly."""
    indexes = Indexes()
    assert len(indexes) == 0


def test_multi_index_auto_detect_from_merge():
    """Bare Indexes() auto-detects key kind from first merge() call."""
    ndim = 10
    index_uuid = Index(ndim=ndim, key_kind="uuid")
    index_uuid.add(b"\x00" * 15 + b"\x01", np.random.rand(ndim).astype(np.float32))

    indexes = Indexes()
    indexes.merge(index_uuid)
    assert len(indexes) == 1


def test_multi_index_path_key_kind_mismatch(tmp_path):
    """Loading a mismatched-key-kind path raises TypeError, not a crash."""
    ndim = 10
    index_uuid = Index(ndim=ndim, key_kind="uuid")
    index_uuid.add(b"\x00" * 15 + b"\x01", np.random.rand(ndim).astype(np.float32))
    uuid_path = str(tmp_path / "uuid_shard.usearch")
    index_uuid.save(uuid_path)

    index_u64 = Index(ndim=ndim, key_kind="u64")
    index_u64.add(42, np.random.rand(ndim).astype(np.float32))
    u64_path = str(tmp_path / "u64_shard.usearch")
    index_u64.save(u64_path)

    # merge_path with explicit key_kind mismatch
    indexes = Indexes(key_kind="u64")
    with pytest.raises(TypeError, match="key_kind"):
        indexes.merge_path(uuid_path)

    # Mixed paths in constructor
    with pytest.raises(TypeError, match="key_kind"):
        Indexes(paths=[uuid_path, u64_path], threads=1)


def test_multi_index_empty_generator_paths():
    """Empty generator as paths does not crash."""
    indexes = Indexes(paths=iter([]))
    assert len(indexes) == 0


def test_kmeans(count_vectors: int = 100, ndim: int = 10, count_clusters: int = 5):
    X = np.random.rand(count_vectors, ndim)
    assignments, distances, centroids = kmeans(X, count_clusters)
    assert len(assignments) == count_vectors
    assert ((assignments >= 0) & (assignments < count_clusters)).all()
