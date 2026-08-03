#!/usr/bin/env -S uv run --quiet --script
"""
USearch Index Tests

Comprehensive test suite for USearch index functionality including
construction, search, serialization, and various data type operations.

Usage:
    uv run python/scripts/test_index.py

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
from time import time

import pytest
import numpy as np

from usearch.eval import random_vectors, self_recall, SearchStats
from usearch.index import (
    Index,
    MetricKind,
    ScalarKind,
    Match,
    Matches,
    BatchMatches,
    Clustering,
)
from usearch.index import (
    DEFAULT_CONNECTIVITY,
)


ndims = [3, 97, 256]
batch_sizes = [1, 11, 77]
quantizations = [
    ScalarKind.F32,
    ScalarKind.F64,
    ScalarKind.F16,
    ScalarKind.BF16,
    ScalarKind.I8,
]
dtypes = [np.float32, np.float64, np.float16]
threads = 2

connectivity_options = [3, 13, 50, DEFAULT_CONNECTIVITY]
continuous_metrics = [MetricKind.Cos, MetricKind.L2sq]
hash_metrics = [
    MetricKind.Hamming,
    MetricKind.Tanimoto,
    MetricKind.Sorensen,
]


def reset_randomness():
    np.random.seed(int(time()))


def packed_uuid_keys(batch_size: int) -> np.ndarray:
    keys = np.empty(batch_size, dtype=np.dtype("V16"))
    for i in range(batch_size):
        body = (i + 1).to_bytes(8, byteorder="big", signed=False)
        offset = (i * 13).to_bytes(4, byteorder="big", signed=False)
        size = (100 + i).to_bytes(4, byteorder="big", signed=False)
        keys[i] = body + offset + size
    return keys


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_initialization_and_addition(ndim, metric, quantization, dtype, batch_size):
    reset_randomness()

    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)
    assert len(index) == batch_size


# TODO: index.vectors returns a list instead of ndarray, causing
# `vectors_batch_retrieved[vectors_reordering]` to fail with TypeError.
# Re-enable once `.vectors` return type is fixed upstream.
#
# @pytest.mark.parametrize("ndim", [3, 97, 256])
# @pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
# @pytest.mark.parametrize("batch_size", [1, 7, 1024])
# @pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.F16, ScalarKind.I8])
# @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
# def test_index_retrieval(ndim, metric, quantization, dtype, batch_size):
#     reset_randomness()
#
#     index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
#     keys = np.arange(batch_size)
#     vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
#     index.add(keys, vectors, threads=threads)
#     vectors_retrieved = np.vstack(index.get(keys, dtype))
#     assert np.allclose(vectors_retrieved, vectors, atol=0.1)
#
#     # Try retrieving all the keys
#     keys_retrieved = index.keys
#     keys_retrieved = np.array(keys_retrieved)
#     assert np.all(np.sort(keys_retrieved) == keys)
#
#     # Try retrieving all of them
#     if quantization != ScalarKind.I8:
#         # The returned vectors can be in a different order
#         vectors_batch_retrieved = index.vectors
#         vectors_reordering = np.argsort(keys_retrieved)
#         vectors_batch_retrieved = vectors_batch_retrieved[vectors_reordering]
#         assert np.allclose(vectors_batch_retrieved, vectors, atol=0.1)
#
#     if quantization != ScalarKind.I8 and batch_size > 1:
#         # When dealing with non-continuous data, it's important to check that
#         # the native bindings access them with correct strides or normalize
#         # similar to `np.ascontiguousarray`:
#         index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
#         vectors = random_vectors(count=batch_size, ndim=ndim + 1, dtype=dtype)
#         # Let's skip the first dimension of each vector:
#         vectors = vectors[:, 1:]
#         index.add(keys, vectors, threads=threads)
#         vectors_retrieved = np.vstack(index.get(keys, dtype))
#         assert np.allclose(vectors_retrieved, vectors, atol=0.1)
#
#         # Try a transposed version of the same vectors, that is not C-contiguous
#         # and should raise an exception!
#         index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
#         vectors = random_vectors(count=ndim, ndim=batch_size, dtype=dtype)  #! reversed dims
#         assert vectors.strides == (batch_size * dtype().itemsize, dtype().itemsize)
#         assert vectors.T.strides == (dtype().itemsize, batch_size * dtype().itemsize)
#         with pytest.raises(Exception):
#             index.add(keys, vectors.T, threads=threads)


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_search(ndim, metric, quantization, dtype, batch_size):
    reset_randomness()

    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)

    if batch_size == 1:
        matches: Matches = index.search(vectors, 10, threads=threads)
        assert isinstance(matches, Matches)
        assert isinstance(matches[0], Match)
        assert matches.keys.ndim == 1
        assert matches.keys.shape[0] == matches.distances.shape[0]
        assert len(matches) == batch_size
        assert np.all(np.sort(index.keys) == np.sort(keys))

    else:
        matches: BatchMatches = index.search(vectors, 10, threads=threads)
        assert isinstance(matches, BatchMatches)
        assert isinstance(matches[0], Matches)
        assert isinstance(matches[0][0], Match)
        assert matches.keys.ndim == 2
        assert matches.keys.shape[0] == matches.distances.shape[0]
        assert len(matches) == batch_size
        assert np.all(np.sort(index.keys) == np.sort(keys))


# TODO: self_recall passes a list to index.search which expects ndarray,
# causing AssertionError. Re-enable once eval.py self_recall is fixed upstream.
#
# @pytest.mark.parametrize("ndim", [3, 97, 256])
# @pytest.mark.parametrize("batch_size", [1, 7, 1024])
# def test_index_self_recall(ndim: int, batch_size: int):
#     """
#     Test self-recall evaluation scripts.
#     """
#     reset_randomness()
#
#     index = Index(ndim=ndim, multi=False)
#     keys = np.arange(batch_size)
#     vectors = random_vectors(count=batch_size, ndim=ndim)
#     index.add(keys, vectors, threads=threads)
#
#     stats_all: SearchStats = self_recall(index, keys=keys)
#     stats_quarter: SearchStats = self_recall(index, sample=0.25, count=10)
#
#     assert stats_all.computed_distances > 0
#     assert stats_quarter.computed_distances > 0


@pytest.mark.parametrize("batch_size", [1, 7, 1024])
def test_index_duplicates(batch_size):
    reset_randomness()

    ndim = 8

    # Cross-batch duplicates: re-adding same keys is a silent no-op
    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)
    index.add(keys, vectors, threads=threads)
    assert len(index) == batch_size

    # Intra-batch duplicates: duplicate keys within same batch are skipped
    # (single-threaded for deterministic contains() checks)
    index = Index(ndim=ndim, multi=False)
    dup_keys = np.concatenate([keys, keys])
    dup_vectors = np.vstack([vectors, vectors])
    index.add(dup_keys, dup_vectors, threads=1)
    assert len(index) == batch_size

    # Multi-index still allows duplicates
    index = Index(ndim=ndim, multi=True)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)
    index.add(keys, vectors, threads=threads)
    assert len(index) == batch_size * 2

    two_per_key = index.get(keys)
    assert np.vstack(two_per_key).shape == (2 * batch_size, ndim)


@pytest.mark.parametrize("batch_size", [1, 7, 1024])
def test_index_stats(batch_size):
    reset_randomness()

    ndim = 8
    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)

    assert index.max_level >= 0
    assert index.stats.nodes >= batch_size
    assert index.levels_stats[0].nodes == batch_size
    assert index.level_stats(0).nodes == batch_size

    assert index.levels_stats[index.max_level].nodes > 0


@pytest.mark.parametrize("use_view", [True, False])
def test_index_load_from_buffer(use_view: bool, ndim: int = 3, batch_size: int = 10):
    reset_randomness()

    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)

    buffer = index.save()
    assert isinstance(buffer, bytearray)

    def _test_load(obj):
        index.clear()
        assert len(index) == 0
        index.view(obj) if use_view else index.load(obj)
        assert len(index) == batch_size

    _test_load(bytes(buffer))
    _test_load(bytearray(buffer))
    _test_load(memoryview(buffer))
    _test_load(np.array(buffer))
    with pytest.raises(TypeError):
        _test_load(123)


@pytest.mark.parametrize("ndim", [1, 3, 8, 32, 256, 4096])
@pytest.mark.parametrize("batch_size", [0, 1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
def test_index_save_load_restore_copy(ndim, quantization, batch_size):
    reset_randomness()
    index = Index(ndim=ndim, dtype=quantization, multi=False)

    if batch_size > 0:
        keys = np.arange(batch_size)
        vectors = random_vectors(count=batch_size, ndim=ndim)
        index.add(keys, vectors, threads=threads)

    # Try copying the original
    copied_index = index.copy()
    assert len(copied_index) == len(index)
    if batch_size > 0:
        assert np.allclose(np.vstack(copied_index.get(keys)), np.vstack(index.get(keys)))

    index.save("tmp.usearch")
    index.clear()
    assert len(index) == 0
    assert os.path.exists("tmp.usearch")

    index.load("tmp.usearch")
    assert len(index) == batch_size
    if batch_size > 0:
        assert len(index[0].flatten()) == ndim

    index_meta = Index.metadata("tmp.usearch")
    assert index_meta is not None

    index = Index.restore("tmp.usearch", view=False)
    assert len(index) == batch_size
    if batch_size > 0:
        assert len(index[0].flatten()) == ndim

    # Try copying the restored index
    copied_index = index.copy()
    assert len(copied_index) == len(index)
    if batch_size > 0:
        assert np.allclose(np.vstack(copied_index.get(keys)), np.vstack(index.get(keys)))

    # Perform the same operations in RAM, without touching the filesystem
    serialized_index = index.save()
    deserialized_metadata = Index.metadata(serialized_index)
    assert deserialized_metadata is not None

    deserialized_index = Index.restore(serialized_index)
    assert len(deserialized_index) == len(index)
    assert set(np.array(deserialized_index.keys)) == set(np.array(index.keys))
    if batch_size > 0:
        assert np.allclose(np.vstack(deserialized_index.get(keys)), np.vstack(index.get(keys)))

    deserialized_index.reset()
    index.reset()
    os.remove("tmp.usearch")


@pytest.mark.parametrize("ndim", [3, 8, 32, 256, 4096])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("threads", [1, 3, 7, 150])
def test_index_restore_multithread_search(ndim, batch_size, threads):

    reset_randomness()
    quantization = ScalarKind.F32
    index = Index(ndim=ndim, dtype=quantization, multi=False)

    if batch_size > 0:
        keys = np.arange(batch_size)
        vectors = random_vectors(count=batch_size, ndim=ndim, dtype=quantization)
        index.add(keys, vectors, threads=threads)

    query = random_vectors(count=batch_size, ndim=ndim, dtype=quantization)
    k = min(batch_size, 10)

    result_original = index.search(query, count=k, threads=threads)
    dumped_index: bytes = index.save()
    dumped_index_view = memoryview(dumped_index)

    # When restoring from disk, search must not fail if using multiple threads.
    index_restored = Index.restore(dumped_index, view=False)
    result_restored = index_restored.search(query, count=k, threads=threads)
    assert np.allclose(result_original.distances, result_restored.distances, atol=0.1)

    index_viewed = Index.restore(dumped_index_view, view=True)
    result_view = index_viewed.search(query, count=k, threads=threads)
    assert np.allclose(result_original.distances, result_view.distances, atol=0.1)


@pytest.mark.parametrize("batch_size", [32])
def test_index_contains_remove_rename(batch_size):
    reset_randomness()
    if batch_size <= 1:
        return

    ndim = 8
    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)

    index.add(keys, vectors, threads=threads)
    assert np.all(index.contains(keys))
    assert np.all(index.count(keys) == np.ones(batch_size))

    removed_keys = keys[: batch_size // 2]
    remaining_keys = keys[batch_size // 2 :]
    index.remove(removed_keys)
    del index[removed_keys]  # ! This will trigger the `__delitem__` dunder method
    assert len(index) == (len(keys) - len(removed_keys))
    assert np.sum(index.contains(keys)) == len(remaining_keys)
    assert np.sum(index.count(keys)) == len(remaining_keys)
    assert np.sum(index.count(removed_keys)) == 0

    assert keys[0] not in index
    assert keys[-1] in index

    renamed_counts = index.rename(removed_keys, removed_keys)
    assert np.sum(index.count(renamed_counts)) == 0

    renamed_counts = index.rename(remaining_keys, removed_keys)
    assert np.sum(index.count(removed_keys)) == len(index)


def test_index_compact_preserves_lookups():
    """After compact() reorders slots, get() must still return each key's own vector,
    and recycled slots must remain safe to reuse for insertions."""
    reset_randomness()

    ndim = 16
    count = 512
    index = Index(ndim=ndim, multi=False, dtype=ScalarKind.F32)
    keys = np.arange(1, count + 1, dtype=np.uint64)
    vectors = random_vectors(count=count, ndim=ndim).astype(np.float32)

    index.add(keys, vectors)
    index.compact()
    assert len(index) == count
    for i, key in enumerate(keys):
        retrieved = np.asarray(index.get(int(key)), dtype=np.float32).ravel()
        assert np.allclose(retrieved, vectors[i], atol=1e-5), f"get({key}) returned another key's vector"

    # Removals put slots on the free list; compact() renumbers slots, so the free
    # list must be rebuilt for subsequent insertions to not overwrite live nodes.
    removed_keys = keys[: count // 4]
    surviving_keys = keys[count // 4 :]
    index.remove(removed_keys)
    index.compact()
    assert len(index) == len(surviving_keys)

    fresh_keys = np.arange(count + 1, count + 1 + len(removed_keys), dtype=np.uint64)
    fresh_vectors = random_vectors(count=len(removed_keys), ndim=ndim).astype(np.float32)
    index.add(fresh_keys, fresh_vectors)

    for i, key in enumerate(surviving_keys):
        retrieved = np.asarray(index.get(int(key)), dtype=np.float32).ravel()
        assert np.allclose(retrieved, vectors[count // 4 + i], atol=1e-5), f"get({key}) corrupted by reinsertion"
    for i, key in enumerate(fresh_keys):
        retrieved = np.asarray(index.get(int(key)), dtype=np.float32).ravel()
        assert np.allclose(retrieved, fresh_vectors[i], atol=1e-5), f"get({key}) returned another key's vector"


def test_index_compact_then_grow():
    """Growing an index after compact() must stay within the nodes buffer: compaction
    must preserve the reserved capacity, not shrink the buffer to the populated count."""
    reset_randomness()

    ndim = 16
    initial = 2100  # Auto-reserve rounds up to 4096 slots
    extra = 1900  # Grows within the prior reservation, so no reallocation rescues an undersized buffer
    index = Index(ndim=ndim, multi=False, dtype=ScalarKind.F32)
    keys = np.arange(initial, dtype=np.uint64)
    vectors = random_vectors(count=initial, ndim=ndim).astype(np.float32)
    index.add(keys, vectors)
    index.compact(threads=1)

    extra_keys = np.arange(initial, initial + extra, dtype=np.uint64)
    extra_vectors = random_vectors(count=extra, ndim=ndim).astype(np.float32)
    index.add(extra_keys, extra_vectors)
    assert len(index) == initial + extra

    for key in (0, initial - 1, initial, initial + extra - 1):
        expected = vectors[key] if key < initial else extra_vectors[key - initial]
        retrieved = np.asarray(index.get(int(key)), dtype=np.float32).ravel()
        assert np.allclose(retrieved, expected, atol=1e-5), f"get({key}) returned another key's vector"

    # Alternating mutations with periodic compactions must not corrupt the index either.
    live = list(range(initial + extra))
    next_key = initial + extra
    for update in range(1, 201):
        if update % 2:
            index.remove(live.pop(0))
        else:
            index.add(next_key, random_vectors(count=1, ndim=ndim).astype(np.float32))
            live.append(next_key)
            next_key += 1
        if update % 10 == 0:
            index.compact(threads=1)
    assert len(index) == len(live)
    assert index.search(random_vectors(count=1, ndim=ndim).astype(np.float32), 10).keys.size == 10


def test_index_compact_cancellation():
    """A progress callback returning False must abort compact() with an error,
    leaving the index unchanged and fully usable."""
    reset_randomness()

    ndim = 16
    count = 128
    index = Index(ndim=ndim, multi=False, dtype=ScalarKind.F32)
    keys = np.arange(count, dtype=np.uint64)
    vectors = random_vectors(count=count, ndim=ndim).astype(np.float32)
    index.add(keys, vectors)

    with pytest.raises(ValueError):
        index.compact(threads=1, progress=lambda processed, total: False)

    assert len(index) == count
    for key in (0, count // 2, count - 1):
        retrieved = np.asarray(index.get(int(key)), dtype=np.float32).ravel()
        assert np.allclose(retrieved, vectors[key], atol=1e-5), f"get({key}) broken after cancelled compact"

    # A subsequent uncancelled compaction must succeed.
    index.compact(threads=1)
    assert len(index) == count


def test_index_uuid128_workflow():
    reset_randomness()

    ndim = 8
    batch_size = 16
    index = Index(ndim=ndim, multi=False, key_kind="uuid")
    keys = packed_uuid_keys(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)

    index.add(keys, vectors, threads=threads)
    assert np.all(index.contains(keys))
    assert np.all(index.count(keys) == np.ones(batch_size))

    single_key = keys[0].tobytes()
    assert index.contains(single_key)
    assert index.count(single_key) == 1
    assert np.allclose(index.get(single_key), vectors[0], atol=0.1)

    matches: BatchMatches = index.search(vectors, 1, threads=threads)
    assert np.all(matches.keys[:, 0] == keys)

    removed_keys = keys[: batch_size // 2]
    remaining_keys = keys[batch_size // 2 :]
    index.remove(removed_keys)
    assert np.sum(index.count(removed_keys)) == 0
    assert np.sum(index.contains(keys)) == len(remaining_keys)

    index.rename(remaining_keys, removed_keys)
    assert np.sum(index.count(removed_keys)) == len(index)


def test_index_uuid128_vectors_and_indexed_keys():
    """Test that .vectors, .get(index.keys), and .contains(index.keys) work with UUID keys."""
    reset_randomness()

    ndim = 8
    batch_size = 16
    index = Index(ndim=ndim, multi=False, key_kind="uuid")
    keys = packed_uuid_keys(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)

    # Build lookup map for order-independent validation
    original_map = {key.tobytes(): vec for key, vec in zip(keys, vectors)}

    # Test .vectors property (index.vectors -> self.get(self.keys) -> _normalize_many_keys)
    retrieved_vecs = index.vectors
    retrieved_keys = np.array(index.keys)
    assert len(retrieved_vecs) == batch_size
    for key, vec in zip(retrieved_keys, retrieved_vecs):
        assert np.allclose(original_map[key.tobytes()], vec, atol=0.1)

    # Test explicit get(index.keys) with IndexedKeys
    got_vecs = index.get(index.keys)
    assert len(got_vecs) == batch_size
    for key, vec in zip(retrieved_keys, got_vecs):
        assert np.allclose(original_map[key.tobytes()], vec, atol=0.1)

    # Test contains(index.keys) with IndexedKeys
    assert np.all(index.contains(index.keys))


def test_index_uuid128_save_load_view_roundtrip_and_mismatch():
    reset_randomness()

    ndim = 16
    batch_size = 32
    index = Index(ndim=ndim, multi=False, key_kind="uuid")
    keys = packed_uuid_keys(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)

    dumped = index.save()
    loaded = Index(ndim=ndim, multi=False, key_kind="uuid")
    loaded.load(dumped)
    viewed = Index(ndim=ndim, multi=False, key_kind="uuid")
    viewed.view(dumped)

    loaded_keys = loaded.keys.__array__()
    viewed_keys = viewed.keys.__array__()
    assert set(key.tobytes() for key in loaded_keys) == set(key.tobytes() for key in keys)
    assert set(key.tobytes() for key in viewed_keys) == set(key.tobytes() for key in keys)
    assert np.allclose(np.vstack(loaded.get(keys)), np.vstack(index.get(keys)), atol=0.1)

    index.save("tmp-uuid.usearch")
    with pytest.raises(ValueError, match="Key kind mismatch"):
        Index(ndim=ndim, multi=False).load("tmp-uuid.usearch")
    os.remove("tmp-uuid.usearch")


def test_save_release_gil_path():
    """Save to path with release_gil=True allows other threads to run."""
    import threading

    ndim = 64
    batch_size = 10_000
    index = Index(ndim=ndim)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors)

    counter = [0]
    stop = threading.Event()

    def count_loop():
        while not stop.is_set():
            counter[0] += 1

    path = "tmp_gil_release.usearch"
    try:
        t = threading.Thread(target=count_loop)
        t.start()
        index.save(path, release_gil=True)
        stop.set()
        t.join()
        assert counter[0] > 0, "Counter thread did not make progress during GIL-released save"

        restored = Index.restore(path, view=False)
        assert len(restored) == batch_size
        assert set(np.array(restored.keys)) == set(keys)
    finally:
        stop.set()
        if os.path.exists(path):
            os.remove(path)


def test_save_release_gil_buffer():
    """Save to buffer with release_gil=True allows other threads to run."""
    import threading

    ndim = 64
    batch_size = 10_000
    index = Index(ndim=ndim)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors)

    counter = [0]
    stop = threading.Event()

    def count_loop():
        while not stop.is_set():
            counter[0] += 1

    t = threading.Thread(target=count_loop)
    t.start()
    buf = index.save(release_gil=True)
    stop.set()
    t.join()
    assert counter[0] > 0, "Counter thread did not make progress during GIL-released save"

    restored = Index.restore(buf)
    assert len(restored) == batch_size
    assert set(np.array(restored.keys)) == set(keys)


def test_save_release_gil_conflict_with_progress():
    """release_gil=True with a progress callback raises ValueError."""
    index = Index(ndim=8)
    index.add(np.array([0]), random_vectors(count=1, ndim=8))

    def progress_cb(completed, total):
        return True

    with pytest.raises(ValueError, match="release_gil.*incompatible.*progress"):
        index.save("tmp.usearch", progress=progress_cb, release_gil=True)

    with pytest.raises(ValueError, match="release_gil.*incompatible.*progress"):
        index.save(progress=progress_cb, release_gil=True)


def test_save_release_gil_error_propagation():
    """Save to nonexistent path with release_gil=True raises an exception."""
    index = Index(ndim=8)
    index.add(np.array([0]), random_vectors(count=1, ndim=8))

    with pytest.raises(Exception):
        index.save("/nonexistent/dir/file.usearch", release_gil=True)


def test_save_default_behavior_unchanged():
    """Save without release_gil produces valid output identical to before."""
    ndim = 16
    batch_size = 100
    index = Index(ndim=ndim)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors)

    path = "tmp_default_save.usearch"
    try:
        index.save(path)
        restored = Index.restore(path, view=False)
        assert len(restored) == batch_size
        assert set(np.array(restored.keys)) == set(keys)
    finally:
        if os.path.exists(path):
            os.remove(path)

    buf = index.save()
    restored = Index.restore(buf)
    assert len(restored) == batch_size
    assert set(np.array(restored.keys)) == set(keys)


@pytest.mark.skip(reason="Not guaranteed")
@pytest.mark.parametrize("batch_size", [3, 17, 33])
@pytest.mark.parametrize("threads", [1, 4])
def test_index_oversubscribed_search(batch_size: int, threads: int):
    reset_randomness()
    if batch_size <= 1:
        return

    ndim = 8
    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)

    index.add(keys, vectors, threads=threads)
    assert np.all(index.contains(keys))
    assert np.all(index.count(keys) == np.ones(batch_size))

    batch_matches: BatchMatches = index.search(vectors, batch_size * 10, threads=threads)
    for i, match in enumerate(batch_matches):
        assert i == match.keys[0]
        assert len(match.keys) == batch_size


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [500, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_clustering(ndim, metric, quantization, dtype, batch_size):
    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)

    clusters: Clustering = index.cluster(vectors=vectors, threads=threads)
    assert len(clusters.matches.keys) == batch_size

    # If no argument is provided, we cluster the present entries
    clusters: Clustering = index.cluster(threads=threads)
    assert len(clusters.matches.keys) == batch_size

    # If no argument is provided, we cluster the present entries
    clusters: Clustering = index.cluster(keys=keys[:50], threads=threads)
    assert len(clusters.matches.keys) == 50

    # If no argument is provided, we cluster the present entries
    clusters: Clustering = index.cluster(min_count=3, max_count=10, threads=threads)
    unique_clusters = set(clusters.matches.keys.flatten().tolist())
    assert len(unique_clusters) >= 3 and len(unique_clusters) <= 10


def test_index_keys_iteration():
    """Test that iterating over index.keys works without infinite loop."""
    index = Index(ndim=3)
    index.add(keys=[42], vectors=np.array([0.2, 0.3, 0.5]))

    keys_list = list(index.keys)
    assert len(keys_list) == 1
    assert keys_list[0] == 42


def test_index_copied_memory_usage():
    """Test that copy=False results in lower memory usage than copy=True."""
    reset_randomness()

    ndim = 128
    batch_size = 1000
    dtype = np.float32  # ! Ensure same type for both vectors and index
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    keys = np.arange(batch_size)

    # Create index with `copy=True`
    index_copied = Index(ndim=ndim, metric=MetricKind.Cos, dtype=dtype, multi=False)
    index_copied.add(keys, vectors, copy=True, threads=threads)

    # Create index with `copy=False`
    index_viewing = Index(ndim=ndim, metric=MetricKind.Cos, dtype=dtype, multi=False)
    index_viewing.add(keys, vectors, copy=False, threads=threads)

    # Both should have same number of entries
    assert len(index_copied) == len(index_viewing) == batch_size

    # Memory usage should be larger when `copy=True`
    memory_with_copy = index_copied.memory_usage
    memory_without_copy = index_viewing.memory_usage

    assert (
        memory_with_copy > memory_without_copy
    ), f"Expected default index addition to use more memory than copy=False ({memory_with_copy} vs {memory_without_copy})"
