"""Load benchmark datasets.

Every loader returns the same three arrays:
    base     (N, d)  float32  the vectors we index
    queries  (Q, d)  float32  the vectors we search for
    gt       (Q, K)  int32    for each query, the ids of the true K nearest
                              neighbours in `base` (closest first)
"""

import ftplib
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SIFTSMALL_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"


# ---------- .fvecs / .ivecs file format ----------
# Each vector is stored as: [d as int32][d values]. Values are float32 in
# .fvecs and int32 in .ivecs. Both have 4-byte items, so we read everything
# as int32, drop the first column, and reinterpret the bytes if needed.

def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view(np.float32)


def read_ivecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def write_fvecs(path, x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, d = x.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = x.view(np.int32)
    out.tofile(path)


# ---------- ground truth ----------

def exact_ground_truth(base, queries, k):
    """True k nearest neighbours, computed in float64 with a full sort.

    Deliberately slow and simple: it is the reference that the brute-force
    index (which uses float32 and argpartition) is tested against.
    """
    base64 = base.astype(np.float64)
    gt = np.empty((len(queries), k), dtype=np.int32)
    for i, q in enumerate(queries.astype(np.float64)):
        d = ((base64 - q) ** 2).sum(axis=1)
        gt[i] = np.argsort(d, kind="stable")[:k]
    return gt


# ---------- datasets ----------

def synthetic(n=10_000, n_queries=100, dim=128, k=100, seed=0):
    """Random Gaussian vectors. Needs no download, so tests and CI use it."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    return base, queries, exact_ground_truth(base, queries, k)


def download_siftsmall(data_dir=DATA_DIR):
    """Download SIFT10K (about 5 MB) from the INRIA TEXMEX corpus."""
    data_dir = Path(data_dir)
    target = data_dir / "siftsmall"
    if (target / "siftsmall_base.fvecs").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "siftsmall.tar.gz"
    print(f"downloading {SIFTSMALL_URL}")
    # urllib's FTP handler fails on this server's passive-mode reply; ftplib works.
    url = urlparse(SIFTSMALL_URL)
    with ftplib.FTP(url.hostname, timeout=60) as ftp, open(archive, "wb") as f:
        ftp.login()
        ftp.retrbinary(f"RETR {url.path}", f.write)
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter="data")
    archive.unlink()
    return target


def siftsmall(data_dir=DATA_DIR):
    """SIFT10K: 10,000 base vectors, 100 queries, 128 dims, top-100 ground truth."""
    d = Path(data_dir) / "siftsmall"
    if not (d / "siftsmall_base.fvecs").exists():
        raise FileNotFoundError(
            f"{d} not found. Run: python -m vecsearch.datasets download"
        )
    return (
        read_fvecs(d / "siftsmall_base.fvecs"),
        read_fvecs(d / "siftsmall_query.fvecs"),
        read_ivecs(d / "siftsmall_groundtruth.ivecs"),
    )


LOADERS = {"synthetic": synthetic, "siftsmall": siftsmall}


def load(name):
    return LOADERS[name]()


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["download"]:
        print("saved to", download_siftsmall())
    else:
        print("usage: python -m vecsearch.datasets download")
