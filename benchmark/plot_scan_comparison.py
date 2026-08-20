#!/usr/bin/env python
"""Comparison plot: full-size-scan benchmarks, new implementations vs previous.

Reads three CSVs and produces one log-log figure per model with four series:
previous Julia, new Julia (matrix + matrix-free), new Python (matrix +
matrix-free).  A combined 3-panel figure is also written.

Usage::

    uv run python benchmark/plot_scan_comparison.py
"""

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JULIA_DIR = os.path.join(
    os.path.dirname(HERE), "..", "RealSpace_ExactDiagonalization",
    "benchmark", "benchmark_data",
)
PY_DIR = os.path.join(HERE, "benchmark_data")
FIG_DIR = os.path.join(HERE, "figures_py")

MODELS = ["Heisenberg", "Haldane_Boson", "Hubbard_Fermion"]
MODE_STYLE = {
    "matrix": {"ls": "-", "mk": "o"},
    "matrixfree": {"ls": "--", "mk": "s"},
}


def read_csv(path):
    """Read a benchmark CSV into {(model, label, mode, sector_dim): elapsed}.

    Args:
        path: CSV path.

    Returns:
        dict: keyed by ``(model, label, mode, sector_dim)`` → ``elapsed_s``.
    """
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            out[
                (row["model"], row["label"], row["mode"], int(row["sector_dim"]))
            ] = float(row["elapsed_s"])
    return out


def main():
    import matplotlib.pyplot as plt

    prev = read_csv(os.path.join(JULIA_DIR, "benchmark_raw_v0_previous.csv"))
    new_jl = read_csv(os.path.join(JULIA_DIR, "benchmark_raw.csv"))
    new_py = read_csv(os.path.join(PY_DIR, "benchmark_raw_py_scan.csv"))

    os.makedirs(FIG_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5))
    for ax, model in zip(axes, MODELS):
        ax.set_title(model)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Sector Hilbert-space dimension")
        ax.set_ylabel("Single-sector ED time (s)")

        def plot_series(data, label, color, alpha=1.0, markersize=6):
            pts = sorted(
                (
                    (sector_dim, elapsed)
                    for (m, _, mode, sector_dim), elapsed in data.items()
                    if m == model
                ),
                key=lambda p: p[0],
            )
            if not pts:
                return
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=color, alpha=alpha,
                    markersize=markersize, label=label)

        # previous Julia (historical reference; matrix mode only)
        prev_pts = sorted(
            (
                (sector_dim, elapsed)
                for (m, _, mode, sector_dim), elapsed in prev.items()
                if m == model and mode == "matrix"
            ),
            key=lambda p: p[0],
        )
        if prev_pts:
            xs, ys = zip(*prev_pts)
            ax.plot(xs, ys, "^-", color="0.45", alpha=0.55,
                    markersize=6, label="previous Julia (matrix)")

        for mode, color in (("matrix", "royalblue"), ("matrixfree", "darkorange")):
            plot_series(
                {(m, l, mo, d): e for (m, l, mo, d), e in new_jl.items()
                 if mo == mode},
                f"new Julia {mode}", color, alpha=0.9,
            )
            plot_series(
                {(m, l, mo, d): e for (m, l, mo, d), e in new_py.items()
                 if mo == mode},
                f"new Python {mode}", color, alpha=0.45, markersize=5,
            )

        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "benchmark_scan_new_vs_previous.svg")
    fig.savefig(out)
    print(f"→ {out}")
    plt.close(fig)

    # also a plain new-vs-new figure per model for the notebooks
    fig2, axes2 = plt.subplots(1, 3, figsize=(18.0, 5.5))
    for ax, model in zip(axes2, MODELS):
        ax.set_title(model)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Sector Hilbert-space dimension")
        ax.set_ylabel("Single-sector ED time (s)")
        for mode, color in (("matrix", "royalblue"), ("matrixfree", "darkorange")):
            for name, data, alpha in (
                ("Julia", new_jl, 0.9), ("Python", new_py, 0.45)
            ):
                pts = sorted(
                    (
                        (sector_dim, elapsed)
                        for (m, _, mo, sector_dim), elapsed in data.items()
                        if m == model and mo == mode
                    ),
                    key=lambda p: p[0],
                )
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, "-o", color=color, alpha=alpha,
                            markersize=5, label=f"{name} {mode}")
        ax.legend(loc="upper left", fontsize=8)
    fig2.tight_layout()
    out2 = os.path.join(FIG_DIR, "benchmark_scan_new_py_vs_julia.svg")
    fig2.savefig(out2)
    print(f"→ {out2}")
    plt.close(fig2)

    # print a summary table
    print("\nSummary (sector_dim -> elapsed_s [matrix / matrixfree]):")
    for model in MODELS:
        print(f"  {model}:")
        rows = {}
        for (m, lbl, mode, d), e in new_jl.items():
            if m == model:
                rows.setdefault((d, lbl), {})[f"jl_{mode}"] = e
        for (m, lbl, mode, d), e in new_py.items():
            if m == model:
                rows.setdefault((d, lbl), {})[f"py_{mode}"] = e
        for (d, lbl) in sorted(rows):
            r = rows[(d, lbl)]
            print(
                f"    dim={d:>9} {lbl:<6} "
                f"jl: {r.get('jl_matrix', float('nan')):8.3f}/{r.get('jl_matrixfree', float('nan')):8.3f} s  "
                f"py: {r.get('py_matrix', float('nan')):8.3f}/{r.get('py_matrixfree', float('nan')):8.3f} s"
            )


if __name__ == "__main__":
    main()
