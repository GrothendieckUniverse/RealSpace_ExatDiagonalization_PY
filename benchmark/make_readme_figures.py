#!/usr/bin/env python
"""Generate per-module README benchmark figures.

Each figure is **one module's own data** (not a cross-language comparison):

- ``readme_benchmark_<model>_py.svg``   — Python scan CSV, matrix + matrix-free
- ``readme_benchmark_<model>_jl.svg``   — Julia scan CSV, matrix + matrix-free
- ``python_numba_vs_numpy.svg``          — Numba vs pure-numpy A/B (Python)
- ``julia_distributed_vs_threading.svg`` — distributed vs threads A/B (Julia)

x-axis ticks are the sample size with the sector Hilbert-space dimension
underneath (e.g. ``N=28\\nD=1432860``).
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JULIA_DIR = os.path.join(os.path.dirname(ROOT), "RealSpace_ExactDiagonalization",
                         "benchmark", "benchmark_data")
PY_DIR = os.path.join(HERE, "benchmark_data")
FIG_DIR = os.path.join(HERE, "figures_py")
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = ["Heisenberg", "Haldane_Boson", "Hubbard_Fermion"]
MODEL_TITLES = {
    "Heisenberg": "Heisenberg chain (half filling)",
    "Haldane_Boson": "Bosonic Haldane FCI (half band filling)",
    "Hubbard_Fermion": "Spinful Fermi-Hubbard (half filling)",
}
MODE_STYLE = {"matrix": ("royalblue", "-", "o"),
              "matrixfree": ("darkorange", "--", "s")}


def read_csv(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            out[(row["model"], row["label"], row["mode"])] = (
                float(row["elapsed_s"]), int(row["sector_dim"]))
    return out


def size_order(model):
    def key(lbl):
        if "=" in lbl:          # Heisenberg "N=18"
            return int(lbl.split("=")[1])
        a, b = lbl.split("\u00d7")   # "2×3" etc.
        return int(a) * int(b)
    return key


def plot_one_module(data, module, outfile_template):
    """3 per-model figures from one module's scan CSV."""
    made = []
    for model in MODELS:
        labels = sorted({lbl for (m, lbl, _) in data if m == model},
                        key=size_order(model))
        if not labels:
            continue
        xs = list(range(len(labels)))
        dims = [data[(model, lbl, "matrix")][1] for lbl in labels]

        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        ax.set_yscale("log")
        ax.set_xlabel("Sample size  (sector Hilbert-space dimension below)")
        ax.set_ylabel("Single-sector ED time (s)")
        ax.set_title(f"{MODEL_TITLES[model]} \u2014 {module}")

        for mode, (color, ls, mk) in MODE_STYLE.items():
            ys = [data[(model, lbl, mode)][0] for lbl in labels
                  if (model, lbl, mode) in data]
            xsub = [xs[i] for i, lbl in enumerate(labels)
                    if (model, lbl, mode) in data]
            ax.plot(xsub, ys, linestyle=ls, marker=mk, color=color,
                    markersize=6, label=mode)

        ax.set_xticks(xs)
        ax.set_xticklabels([f"{lbl}\nD={d}" for lbl, d in zip(labels, dims)],
                           fontsize=8)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, outfile_template.format(model=model))
        fig.savefig(out)
        plt.close(fig)
        made.append(out)
    return made


def main():
    py = read_csv(os.path.join(PY_DIR, "benchmark_raw_py_scan.csv"))
    jl = read_csv(os.path.join(JULIA_DIR, "benchmark_raw.csv"))

    print("Python rows:", len(py), " Julia rows:", len(jl))

    for out in plot_one_module(py, "Python (pure numpy)",
                               "readme_benchmark_{model}_py.svg"):
        print("\u2192", out)
    for out in plot_one_module(jl, "Julia (multithreaded)",
                               "readme_benchmark_{model}_jl.svg"):
        print("\u2192", out)

    # \u2500\u2500 numba vs numpy A/B (Python) \u2500\u2500
    kernels = ["orbit catalog", "H-block", "projection table", "matvec (\u00d71e3)"]
    numba_t = [1.236, 0.227, 0.234, 4.15]
    numpy_t = [1.209, 0.127, 0.123, 2.61]
    xpos = range(len(kernels))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.35
    b1 = ax.bar([x - w / 2 for x in xpos], numba_t, w, label="Numba (removed)",
                color="#d62728")
    b2 = ax.bar([x + w / 2 for x in xpos], numpy_t, w, label="pure numpy (adopted)",
                color="#1f77b4")
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(kernels)
    ax.set_ylabel("time (s; matvec in ms)")
    ax.set_title("Python hot kernels: Numba vs pure numpy (Heis N=24, identical results)")
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.2f}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "python_numba_vs_numpy.svg")
    fig.savefig(out)
    plt.close(fig)
    print("\u2192", out)

    # \u2500\u2500 julia distributed vs threading A/B \u2500\u2500
    sizes = ["N=18", "N=20", "N=22", "N=24", "N=26"]
    dist_mat = [10.78, 10.37, 11.38, 15.79, 27.08]
    thr_mat = [0.28, 0.31, 0.49, 2.15, 9.68]
    dist_mf = [3.60, 4.29, 11.70, 30.76, None]   # N=26 matrix-free: killed (OOM)
    thr_mf = [0.21, 0.24, 0.74, 2.52, 12.83]
    xs = range(len(sizes))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.set_yscale("log")
    ax.plot(xs, dist_mat, "-o", color="#d62728",
            label="distributed -p 8 (matrix, removed)")
    ax.plot(xs, thr_mat, "-s", color="#1f77b4", label="Threads.@threads -t 10 (matrix)")
    ax.plot(xs, dist_mf, "--o", color="#ff9896",
            label="distributed -p 8 (matrix-free, removed)")
    ax.plot(xs, thr_mf, "--s", color="#9ecae1",
            label="Threads.@threads -t 10 (matrix-free)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Heisenberg chain size")
    ax.set_ylabel("Single-sector ED time (s)")
    ax.set_title("Julia: distributed vs multithreading \u2014 distributed died at N=26/28 (OOM)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "julia_distributed_vs_threading.svg")
    fig.savefig(out)
    plt.close(fig)
    print("\u2192", out)

    print("README FIGURES DONE")


if __name__ == "__main__":
    main()
