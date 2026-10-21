import numpy as np

from bench.run import measure
from vecsearch import IVFIndex
from vecsearch.datasets import synthetic


def built(nlist=20):
    base, queries, gt = synthetic(n=2000, n_queries=50, dim=16, k=10)
    index = IVFIndex(dim=16, nlist=nlist)
    index.add(base)
    return index, queries, gt


def test_every_vector_in_exactly_one_list():
    index, _, _ = built()
    all_ids = np.concatenate(index.list_ids)
    assert sorted(all_ids.tolist()) == list(range(2000))


def test_probing_every_list_is_exact():
    index, queries, gt = built()
    index.nprobe = index.nlist
    recall, _ = measure(index, queries, gt, 10, repeats=1)
    assert recall == 1.0


def test_recall_never_drops_when_nprobe_grows():
    # probing the nearest p lists is a subset of probing the nearest p+1,
    # so recall can only stay the same or go up
    index, queries, gt = built()
    recalls = []
    for p in (1, 2, 4, 8, 16):
        index.nprobe = p
        recalls.append(measure(index, queries, gt, 10, repeats=1)[0])
    assert recalls == sorted(recalls)
    assert recalls[0] < 1.0  # nprobe=1 really is approximate here


def test_add_twice_keeps_ids():
    index = IVFIndex(dim=2, nlist=2)
    index.add([[0, 0], [10, 10], [0, 1], [10, 11]])
    index.add([[0.2, 0.2]])
    index.nprobe = 2
    ids, _ = index.search(np.array([0.2, 0.2], dtype=np.float32), k=1)
    assert ids.tolist() == [4]
    assert len(index) == 5


def test_memory_counts_vectors_ids_norms_and_centers():
    index, _, _ = built(nlist=20)
    assert index.memory_bytes() == 2000 * 16 * 4 + 2000 * 8 + 2000 * 4 + 20 * 16 * 4
