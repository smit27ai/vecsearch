import numpy as np
import pytest

from bench.run import measure
from vecsearch import IVFPQIndex, PQIndex, ProductQuantizer
from vecsearch.datasets import synthetic


@pytest.fixture(scope="module")
def data():
    return synthetic(n=2000, n_queries=30, dim=16, k=10)


# ---------- ProductQuantizer ----------

def test_codes_are_bytes(data):
    base, _, _ = data
    pq = ProductQuantizer(16, m=4, ksub=64)
    pq.train(base)
    codes = pq.encode(base)
    assert codes.shape == (2000, 4) and codes.dtype == np.uint8


def test_exact_when_every_point_has_its_own_center():
    # 16 points, 16 centers per chunk: every chunk maps to itself
    x = np.random.default_rng(0).standard_normal((16, 8)).astype(np.float32)
    pq = ProductQuantizer(8, m=2, ksub=16)
    pq.train(x)
    np.testing.assert_allclose(pq.decode(pq.encode(x)), x, rtol=1e-6)


def test_adc_equals_distance_to_decoded_vector(data):
    # ADC is not "roughly" the distance: it is exactly the distance from the
    # query to the decoded (approximate) vector
    base, queries, _ = data
    pq = ProductQuantizer(16, m=4, ksub=64)
    pq.train(base)
    codes = pq.encode(base)
    q = queries[0]
    adc = pq.adc(pq.distance_table(q), codes.T)
    exact_to_decoded = ((pq.decode(codes) - q) ** 2).sum(1)
    np.testing.assert_allclose(adc, exact_to_decoded, rtol=1e-4, atol=1e-3)


def test_more_chunks_less_error(data):
    base, _, _ = data
    errors = []
    for m in (2, 4, 8):
        pq = ProductQuantizer(16, m=m, ksub=64)
        pq.train(base)
        errors.append(((pq.decode(pq.encode(base)) - base) ** 2).sum(1).mean())
    assert errors == sorted(errors, reverse=True)


def test_bad_settings():
    with pytest.raises(ValueError):
        ProductQuantizer(10, m=4)  # 10 not divisible by 4
    with pytest.raises(ValueError):
        ProductQuantizer(16, m=4, ksub=512)  # codes would not fit in a byte


# ---------- PQIndex ----------

def test_pq_index_recall_and_rerank(data):
    base, queries, gt = data
    plain = PQIndex(16, m=4, ksub=64)
    plain.add(base)
    reranked = PQIndex(16, m=4, ksub=64, keep_vectors=True, rerank=100)
    reranked.add(base)
    r_plain = measure(plain, queries, gt, 10, repeats=1)[0]
    r_rerank = measure(reranked, queries, gt, 10, repeats=1)[0]
    assert 0.2 < r_plain < r_rerank
    assert r_rerank >= 0.9


def test_rerank_without_vectors_fails(data):
    base, queries, _ = data
    index = PQIndex(16, m=4, ksub=64, rerank=50)
    index.add(base)
    with pytest.raises(ValueError):
        index.search(queries[0], 10)


def test_pq_memory(data):
    base, _, _ = data
    index = PQIndex(16, m=4, ksub=64)
    index.add(base)
    codes, codebooks = 2000 * 4, 4 * 64 * 4 * 4  # m*ksub*dsub float32
    assert index.memory_bytes() == codes + codebooks


# ---------- IVFPQIndex ----------

def test_ivfpq_every_vector_in_one_list(data):
    base, _, _ = data
    index = IVFPQIndex(16, nlist=10, m=4, ksub=64)
    index.add(base)
    assert sorted(np.concatenate(index.list_ids).tolist()) == list(range(2000))
    assert sum(c.shape[1] for c in index.list_codes) == 2000


def test_ivfpq_distance_is_to_center_plus_decoded_residual(data):
    base, queries, _ = data
    index = IVFPQIndex(16, nlist=10, m=4, ksub=64, nprobe=10)
    index.add(base)
    q = queries[0]
    ids, d = index.search(q, 5)
    for i, di in zip(ids, d):
        c = next(c for c in range(10) if i in index.list_ids[c])
        pos = int(np.where(index.list_ids[c] == i)[0][0])
        approx = index.centers[c] + index.pq.decode(index.list_codes[c][:, pos:pos + 1].T)[0]
        np.testing.assert_allclose(di, ((approx - q) ** 2).sum(), rtol=1e-4, atol=1e-3)


def test_ivfpq_recall_never_drops_with_nprobe(data):
    base, queries, gt = data
    index = IVFPQIndex(16, nlist=10, m=4, ksub=64)
    index.add(base)
    recalls = []
    for p in (1, 2, 5, 10):
        index.nprobe = p
        recalls.append(measure(index, queries, gt, 10, repeats=1)[0])
    assert recalls == sorted(recalls)


def test_residuals_beat_raw_vectors(data):
    # same bytes per vector; encoding residuals should be more accurate
    base, queries, gt = data
    flat = PQIndex(16, m=4, ksub=64)
    flat.add(base)
    ivfpq = IVFPQIndex(16, nlist=10, m=4, ksub=64, nprobe=10)  # probe all: only encoding differs
    ivfpq.add(base)
    assert measure(ivfpq, queries, gt, 10, repeats=1)[0] > measure(flat, queries, gt, 10, repeats=1)[0]
