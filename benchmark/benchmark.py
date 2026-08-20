#!/usr/bin/env python
"""Single-sector symmetry-resolved ED benchmark.

Port of ``benchmark/benchmark.jl`` from ``RealSpace_ExactDiagonalization.jl``.

Three models are benchmarked in both matrix-construct and matrix-free
modes:

1. Spin-½ Heisenberg chain  (N = [18, 20, 22, 24, 26]);
2. Bosonic Haldane honeycomb FCI  (sample_size = [[2,3], [2,4], [2,5],
   [3,4], [2,7]]);
3. Spinful Fermi-Hubbard on the square lattice  (sample_size = [[2,2],
   [2,3], [2,4], [2,5], [3,4]]).

Size caps (deliberate): systems whose single-sector ED exceeds ~10 min
are excluded (Heisenberg N ≤ 26, Haldane ≤ [2,7], Hubbard ≤ [3,4]).

The CanonicalMap cache is used uniformly across all modes for O(1)
canonical lookups.  Each measurement: JIT-warmup → time ONE sector
(sector index 0) → CSV output → matplotlib figures.

Usage::

    uv run python benchmark/benchmark.py [--small]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

import realspace_exactdiagonalization_py as ed

OUTDIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(OUTDIR, "benchmark_data")
FIG_DIR = os.path.join(OUTDIR, "figures_py")
MODES = ["matrix", "matrixfree"]
SECTOR_IDX = 0   # only benchmark the first sector
NEV = 1          # we only need the lowest eigenvalue

# Single representative size per model (the timely default benchmark)
SINGLE_SIZES = {
    "Heisenberg": [24],
    "Haldane_Boson": [[3, 4]],
    "Hubbard_Fermion": [[2, 5]],
}

# Full multi-size scan used by --scan (scaling figures)
SCAN_SIZES = {
    "Heisenberg": [18, 20, 22, 24, 26, 28],
    "Haldane_Boson": [[2, 3], [2, 4], [2, 5], [3, 4], [2, 7], [4, 4]],
    "Hubbard_Fermion": [[2, 2], [2, 3], [2, 4], [2, 5], [3, 4], [2, 7]],
}

# Reduced subset used by --small (fast sanity scan)
SMALL_SIZES = {
    "Heisenberg": [18, 20, 22],
    "Haldane_Boson": [[2, 3], [2, 4], [2, 5]],
    "Hubbard_Fermion": [[2, 2], [2, 3], [2, 4]],
}

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


@dataclass
class BenchRow:
    """One benchmark measurement (mirrors the Julia ``BenchRow``)."""

    model: str
    label: str
    n_site: int
    n_filled: int
    full_dim: int
    n_orbits: int
    n_group: int
    sector_dim: int
    mode: str
    elapsed_s: float
    energy: float


# ═════════════════════════════════════════════════════════════════════════════
# Model builders (shared with the examples; see models/{heisenberg,hubbard}.py)
# ═════════════════════════════════════════════════════════════════════════════


def build_heisenberg_ed(N: int):
    """ED data of the Heisenberg chain (half filling, PBC)."""
    return ed.build_heisenberg_ed_data(N)


def build_haldane_ed(sample_size: list[int]):
    """ED data of the Haldane honeycomb FCI (t'' = -0.58)."""
    return ed.build_bosonic_fci_ed_data(sample_size=sample_size)


def build_hubbard_ed(sample_size: list[int]):
    """ED data of the spinful Fermi-Hubbard model on the square lattice."""
    return ed.build_spinful_hubbard_ed_data(sample_size)


# ═════════════════════════════════════════════════════════════════════════════
# Benchmark helpers
# ═════════════════════════════════════════════════════════════════════════════


def time_single_sector(ed_data, mode: str, sector_index: int = SECTOR_IDX):
    """Run a single-sector ED; return ``(elapsed_s, sector_dim, lowest_energy)``.

    Args:
        ed_data: the ED data structure (orbit catalog built outside the
            timed region, exactly as in the Julia benchmark).
        mode: ``"matrix"`` or ``"matrixfree"``.
        sector_index: **0-based** sector index to time.

    Returns:
        tuple[float, int, float]: elapsed seconds, sector dimension, and
        the lowest eigenvalue.
    """
    irrep = ed_data.irrep_list[sector_index]
    t0 = time.perf_counter()
    if mode == "matrix":
        vals, _ = ed.ed_scan_at_irrep_matrix(irrep.label, ed_data, nev=NEV)
    elif mode == "matrixfree":
        vals, _ = ed.ed_scan_at_irrep_matrixfree(irrep.label, ed_data, nev=NEV)
    else:
        raise ValueError(f"unknown mode {mode}")
    elapsed = time.perf_counter() - t0
    dim = ed_data.sector_dims[sector_index]
    return elapsed, dim, float(vals[0])


def warmup() -> None:
    """JIT warmup — run a small system twice in each mode."""
    print("=== JIT Warmup ===\n")
    for mode in MODES:
        for rep in range(2):
            ed_data = build_haldane_ed([2, 3])
            time_single_sector(ed_data, mode)
            print(f"  warmup haldane [2,3] mode={mode} rep={rep + 1}/2 done")
    print("=== Warmup complete ===\n")


def run_benchmarks(
    scan: bool = False, reps: int = 3
) -> list[BenchRow]:
    """Run the benchmark: one size per model (default) or the full scan.

    Args:
        scan: whether to run the full multi-size scan instead of the
            single representative sizes.
        reps: number of timing repetitions per (model, mode) — the
            reported ``elapsed_s`` is the MEAN over the repetitions
            (first rep after warmup excluded from the mean when
            ``reps > 1``).

    Returns:
        list[BenchRow]: one row per (model, size, mode).
    """
    warmup()
    rows: list[BenchRow] = []

    def _size_set(model: str):
        return SCAN_SIZES[model] if scan else SINGLE_SIZES[model]

    def _timed_row(model_name, label, builder, size, mode):
        times = []
        e0 = float("nan")
        dim = 0
        ed_data = None
        for rep in range(reps):
            ed_data = builder(size)
            t_elapsed, dim, e0 = time_single_sector(ed_data, mode)
            times.append(t_elapsed)
        used = times[1:] if reps > 1 else times  # drop the first rep
        mean_t = sum(used) / len(used)
        n_site = ed_data.second_quantized_model.lattice.n_site
        n_filled = ed_data.n_filled
        row = BenchRow(
            model_name, label, n_site, n_filled, math.comb(n_site, n_filled),
            len(ed_data.orbit_catalog.representative_mask_list),
            len(ed_data.symmetry_group.operations), dim, mode, mean_t, e0,
        )
        rows.append(row)
        print(
            f"  {label}  mode={mode:<11}  dim={dim}  "
            f"t={round(mean_t, 4)}s (mean of {len(used)} reps)"
        )
        return row

    # ---- 1. Heisenberg chain ----
    print("\n" + "=" * 70)
    print("  Model 1: Spin-½ Heisenberg Chain (translation symmetry)")
    print("=" * 70)
    for N in _size_set("Heisenberg"):
        for mode in MODES:
            _timed_row("Heisenberg", f"N={N}", build_heisenberg_ed, N, mode)

    # ---- 2. Bosonic Haldane FCI ----
    print("\n" + "=" * 70)
    print("  Model 2: Bosonic Haldane FCI (translation symmetry)")
    print("=" * 70)
    for ss in _size_set("Haldane_Boson"):
        for mode in MODES:
            _timed_row(
                "Haldane_Boson", f"{ss[0]}×{ss[1]}", build_haldane_ed, ss, mode
            )

    # ---- 3. Fermionic Hubbard ----
    print("\n" + "=" * 70)
    print("  Model 3: Spinful Fermi-Hubbard (translation symmetry)")
    print("=" * 70)
    for ss in _size_set("Hubbard_Fermion"):
        for mode in MODES:
            _timed_row(
                "Hubbard_Fermion", f"{ss[0]}×{ss[1]}", build_hubbard_ed, ss, mode
            )

    return rows


def write_csv(path: str, rows: list[BenchRow]) -> None:
    """Write the benchmark rows as CSV (same columns as the Julia CSV).

    Args:
        path: output path.
        rows: benchmark measurements.
    """
    fieldnames = [
        "model", "label", "n_site", "n_filled", "full_dim", "n_orbits",
        "n_group", "sector_dim", "mode", "elapsed_s", "energy",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            d = asdict(r)
            d["elapsed_s"] = f"{r.elapsed_s:.9f}"
            d["energy"] = f"{r.energy:.15f}"
            writer.writerow(d)


def _load_julia_rows() -> dict[str, list]:
    """Load the Julia reference CSV (same directory, if present).

    Returns:
        dict[str, list]: mapping ``model_label_mode → elapsed_s``.
    """
    ref_path = os.path.join(RESULT_DIR, "benchmark_raw.csv")
    if not os.path.isfile(ref_path):
        # fall back to the Julia repo's reference CSVs (this machine's runs):
        # `benchmark_raw.csv` (historical multi-size) then the single-size
        # `benchmark_raw_single.csv`.
        julia_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "RealSpace_ExactDiagonalization",
            "benchmark", "benchmark_data",
        )
        for name in ("benchmark_raw.csv", "benchmark_raw_single.csv"):
            candidate = os.path.join(julia_dir, name)
            if os.path.isfile(candidate):
                ref_path = candidate
                break
        else:
            return {}
    out = {}
    with open(ref_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[
                (row["model"], row["label"], row["mode"], int(row["sector_dim"]))
            ] = float(row["elapsed_s"])
    return out


def plot_model(rows: list[BenchRow], model_name: str, outfile: str) -> None:
    """Per-model benchmark plot (time vs system size, log y).

    Args:
        rows: benchmark measurements.
        model_name: model to plot.
        outfile: output ``.svg`` path.
    """
    import matplotlib.pyplot as plt

    rs = [r for r in rows if r.model == model_name]
    if not rs:
        return

    labels_mat = list(dict.fromkeys(r.label for r in rs if r.mode == "matrix"))
    xs = list(range(len(labels_mat)))
    colors = {"matrix": "royalblue", "matrixfree": "darkorange"}
    mode_labels = {"matrix": "matrix construction", "matrixfree": "matrix-free"}
    dim_by_label = {
        r.label: r.sector_dim for r in rs if r.mode == "matrix"
    }
    tick_lbls = [f"{lbl}\nD={dim_by_label[lbl]}" for lbl in labels_mat]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.set_xlabel("System size (sector dimension below)")
    ax.set_ylabel("Single-sector ED time (s)")
    ax.set_title(f"{model_name} — One-Sector Benchmark")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_lbls)
    for mode in ("matrix", "matrixfree"):
        mrs = sorted(
            (r for r in rs if r.mode == mode),
            key=lambda r: labels_mat.index(r.label),
        )
        ys = [r.elapsed_s for r in mrs]
        ax.plot(xs, ys, "o-", color=colors[mode], label=mode_labels[mode])
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outfile)
    print(f"  → {outfile}")
    plt.close(fig)


def plot_scaling(rows: list[BenchRow], model_name: str, outfile: str) -> None:
    """Scaling plot (time vs sector dimension, log-log).

    Args:
        rows: benchmark measurements.
        model_name: model to plot.
        outfile: output ``.svg`` path.
    """
    import matplotlib.pyplot as plt

    rs = [r for r in rows if r.model == model_name]
    if not rs:
        return
    colors = {"matrix": "royalblue", "matrixfree": "darkorange"}
    mode_labels = {"matrix": "matrix construction", "matrixfree": "matrix-free"}

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.set_xlabel("Sector Hilbert-space dimension")
    ax.set_ylabel("Single-sector ED time (s)")
    ax.set_title(f"{model_name} — Scaling with Sector Dimension")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for mode in ("matrix", "matrixfree"):
        mrs = sorted((r for r in rs if r.mode == mode), key=lambda r: r.sector_dim)
        ax.plot(
            [r.sector_dim for r in mrs],
            [r.elapsed_s for r in mrs],
            "o-",
            color=colors[mode],
            label=mode_labels[mode],
        )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outfile)
    print(f"  → {outfile}")
    plt.close(fig)


def plot_comparison(rows: list[BenchRow], outfile: str) -> None:
    """Overlay the Python timings with the Julia reference CSV.

    Args:
        rows: python benchmark measurements.
        outfile: output ``.svg`` path.
    """
    import matplotlib.pyplot as plt

    julia = _load_julia_rows()
    if not julia:
        print("  (no Julia reference CSV found — skipping comparison plot)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    for ax, model in zip(axes, ["Heisenberg", "Haldane_Boson", "Hubbard_Fermion"]):
        rs = [r for r in rows if r.model == model]
        if not rs:
            continue
        ax.set_title(model)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Sector dimension")
        ax.set_ylabel("Single-sector ED time (s)")
        for mode, color in (("matrix", "royalblue"), ("matrixfree", "darkorange")):
            mrs = sorted((r for r in rs if r.mode == mode), key=lambda r: r.sector_dim)
            ax.plot(
                [r.sector_dim for r in mrs],
                [r.elapsed_s for r in mrs],
                "o-",
                color=color,
                label=f"Python {mode}",
            )
            jx, jy = [], []
            for r in mrs:
                key = (model, r.label, mode, r.sector_dim)
                if key in julia:
                    jx.append(r.sector_dim)
                    jy.append(julia[key])
            ax.plot(jx, jy, "s--", color=color, alpha=0.45, label=f"Julia {mode}")
        ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(outfile)
    print(f"  → {outfile}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan", action="store_true",
        help="run the full multi-size scan (for scaling figures) instead "
        "of the single representative sizes",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="restrict a --scan run to the reduced system-size subset",
    )
    parser.add_argument(
        "--reps", type=int, default=3,
        help="timing repetitions per (model, mode); mean reported "
        "(default 3)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Single-Sector ED Benchmark (Python port)")
    print("=" * 70)

    rows = run_benchmarks(scan=args.scan or args.small, reps=args.reps)

    suffix = "_scan" if (args.scan or args.small) else ""
    raw_path = os.path.join(RESULT_DIR, f"benchmark_raw_py{suffix}.csv")
    write_csv(raw_path, rows)
    print(f"\nResults → {raw_path}")

    latest_path = os.path.join(RESULT_DIR, "benchmark_raw_latest_py.csv")
    write_csv(latest_path, rows)

    print("\n--- Generating plots ---")
    plot_model(rows, "Heisenberg", os.path.join(FIG_DIR, "heisenberg_1D.svg"))
    plot_model(rows, "Haldane_Boson", os.path.join(FIG_DIR, "bose_hubbard_2D.svg"))
    plot_model(rows, "Hubbard_Fermion", os.path.join(FIG_DIR, "spinful_fermi_hubbard_2D.svg"))
    plot_scaling(rows, "Heisenberg", os.path.join(FIG_DIR, "heisenberg_1D_scaling.svg"))
    plot_scaling(rows, "Haldane_Boson", os.path.join(FIG_DIR, "bose_hubbard_2D_scaling.svg"))
    plot_scaling(rows, "Hubbard_Fermion", os.path.join(FIG_DIR, "spinful_fermi_hubbard_2D_scaling.svg"))
    plot_comparison(rows, os.path.join(FIG_DIR, "benchmark_py_vs_julia.svg"))

    print("\nDone. All plots saved to", FIG_DIR)


if __name__ == "__main__":
    main()
