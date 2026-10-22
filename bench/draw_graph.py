"""Draw a small 2D HNSW graph, one panel per layer, with a search path.

Each panel shows one layer's nodes and links. The red line is a greedy
search for the query (star): it starts at the entry point on the top layer,
moves to whichever neighbour is closer to the query until none is, then
drops a layer and continues from the same node.

Usage:
    python -m bench.draw_graph     # writes docs/hnsw_layers.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vecsearch import HNSWIndex

OUT = Path(__file__).resolve().parent.parent / "docs" / "hnsw_layers.png"


def greedy_path(index, q):
    """[(layer, node)] visited by a greedy walk from the top layer down."""
    d = lambda i: float(((index.data[i] - q) ** 2).sum())
    node, path = index.entry, []
    for layer in range(index.max_level, -1, -1):
        path.append((layer, node))
        while True:
            best = min(index.graph[layer][node], key=d, default=node)
            if d(best) >= d(node):
                break
            node = best
            path.append((layer, node))
    return path


def main(n=400, M=4, seed=3):
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2)).astype(np.float32)
    index = HNSWIndex(2, M=M, ef_construction=32, seed=seed)
    index.add(points)
    q = np.array([0.85, 0.15], dtype=np.float32)
    path = greedy_path(index, q)

    layers = list(range(index.max_level, -1, -1))
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 4.6))
    for ax, layer in zip(np.atleast_1d(axes), layers):
        g = index.graph[layer]
        for a, links in g.items():
            for b in links:
                ax.plot(*zip(points[a], points[b]), color="0.75", lw=0.5, zorder=1)
        ax.scatter(*points.T, s=3, color="0.85", zorder=0)  # all points, faint
        nodes = np.array(list(g))
        ax.scatter(*points[nodes].T, s=12, color="tab:blue", zorder=2)
        steps = [points[i] for l, i in path if l == layer]
        ax.plot(*np.array(steps).T, color="tab:red", lw=2, marker="o", ms=4, zorder=3)
        ax.scatter(*q, marker="*", s=220, color="black", zorder=4)
        ax.set_title(f"layer {layer}: {len(g)} nodes, {len(steps) - 1} steps")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle(f"HNSW on {n} random 2D points (M={M}). Red = greedy search toward the star.")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
