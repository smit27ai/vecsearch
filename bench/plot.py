"""Draw the benchmark charts from a results file.

Left:  recall@k vs queries per second (log). Best is top-right.
       Measures this implementation, Python overhead included.
Middle: recall@k vs distance computations per query (log). Best is
       bottom-right. Measures the algorithm, independent of language.
Right: recall@k vs bytes of RAM per stored vector (log). Best is
       bottom-right.

Each line is one built index; each dot on it is one search setting
(e.g. nprobe, ef_search), labelled with its value.

Usage:
    python -m bench.plot                      # bench/results/synthetic.json
    python -m bench.plot --dataset siftsmall
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, no window
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def label(row):
    build = ", ".join(f"{k}={v}" for k, v in row["build"].items())
    return f"{row['index']} ({build})" if build else row["index"]


def plot(results_path, out_path):
    data = json.loads(Path(results_path).read_text())
    rows = data["results"]

    lines = {}  # one line per (index, build setting)
    for row in rows:
        lines.setdefault(label(row), []).append(row)

    k = rows[0]["k"]
    panels = [
        ("qps", "queries per second, log  (higher = faster)", "Speed of this Python code"),
        ("dist_per_query", "distance computations per query, log  (lower = less work)",
         "Work done by the algorithm"),
        ("bytes_per_vector", "bytes of RAM per vector, log  (lower = smaller)",
         "Memory"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    colors = plt.get_cmap("tab20").colors[::2] + plt.get_cmap("tab20").colors[1::2]
    # one marker per index type; dashed lines = compressed (PQ) indexes
    style = {"ivf": ("o", "-"), "hnsw": ("s", "-"), "pq": ("^", "--"),
             "pq_rerank": ("v", "--"), "ivfpq": ("D", "--")}
    for ax, (key, ylabel, title) in zip(axes, panels):
        line_no = 0
        for name, pts in lines.items():
            pts = sorted((r for r in pts if key in r), key=lambda r: r["recall"])
            if not pts:
                continue
            xs = [r["recall"] for r in pts]
            ys = [r[key] for r in pts]
            if name == "brute":
                ax.scatter(xs, ys, marker="*", s=180, zorder=3, color="black", label=name)
            else:
                marker, ls = style.get(pts[0]["index"], ("o", "-"))
                ax.plot(xs, ys, marker=marker, ls=ls, ms=5,
                        color=colors[line_no % len(colors)], label=name)
                line_no += 1
            for r in pts:
                text = ", ".join(str(v) for v in r["search"].values())
                if text:
                    ax.annotate(text, (r["recall"], r[key]), textcoords="offset points",
                                xytext=(4, 4), fontsize=7, alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel(f"recall@{k}  (higher = more correct)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{data['dataset']}  (dot labels = search setting: nprobe for IVF and "
                 f"IVF-PQ, ef_search for HNSW, rerank for PQ)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic")
    args = parser.parse_args()
    src = RESULTS_DIR / f"{args.dataset}.json"
    out = RESULTS_DIR / f"{args.dataset}.png"
    plot(src, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
