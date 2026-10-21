"""Draw the recall vs QPS chart from a results file.

    x axis: recall@k (right = more correct)
    y axis: queries per second, log scale (up = faster)

Best is the top-right corner. Each line is one built index; each dot on it
is one search setting (e.g. nprobe), labelled with its value.

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

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, pts in lines.items():
        pts = sorted(pts, key=lambda r: r["recall"])
        xs = [r["recall"] for r in pts]
        ys = [r["qps"] for r in pts]
        if len(pts) == 1:
            ax.scatter(xs, ys, marker="*", s=180, zorder=3, color="black", label=name)
        else:
            ax.plot(xs, ys, marker="o", label=name)
        for r in pts:
            text = ", ".join(str(v) for v in r["search"].values())
            if text:
                ax.annotate(text, (r["recall"], r["qps"]), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, alpha=0.7)

    k = rows[0]["k"]
    ax.set_yscale("log")
    ax.set_xlabel(f"recall@{k}  (higher = more correct)")
    ax.set_ylabel("queries per second, log scale  (higher = faster)")
    ax.set_title(f"Recall vs speed on {data['dataset']}  (dot labels = search setting)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
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
