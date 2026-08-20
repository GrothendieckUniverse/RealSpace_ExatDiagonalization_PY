#!/usr/bin/env python
"""Example 4: Exotic fermionic fractional Chern insulator on the checkerboard lattice.

The spinless-fermion checkerboard model with staggered flux has two flat
bands with Chern numbers ±1 [K. Sun, Z. Gu, H. Katsura, S. Das Sarma,
arXiv:1012.5864].  The fine-tuned "exotic" parameters (customized
V1, V2, V3, all scaled by λ = 2.8) at t'' = −0.2 place the model in the
fermionic FCI phase: at ν = 1/3 filling of the lower band (5 fermions on
the 3×5×2 = 30 flattened vertices) the ground state is a fermionic
fractional Chern insulator with three nearly-degenerate ground states on
the torus (GSD = 3).

Usage::

    uv run python examples/fermion_fci_checkerboard.py
"""

from __future__ import annotations

import os

from fractions import Fraction

import realspace_exactdiagonalization_py as ed


def main() -> None:
    sample_size = [3, 5]
    params = dict(ed.models.fermionic_fci.my_optimal_param)
    params["t''"] = -0.2  # → FCI phase
    filling_fraction_per_band = Fraction(1, 3)  # ν = 1/3 per band

    # The phase-explore builder assembles the model and the symmetry-
    # resolved ED data in one call and returns (model, ed_data, n_filled,
    # filling_fraction).
    model, ed_data, n_filled, filling_fraction = (
        ed.build_phase_explore_fermionic_checkerboard_model(
            params, sample_size, filling_fraction_per_band
        )
    )

    ed.ed_scan(ed_data, nev=5, mode="matrix")

    all_vals: list[float] = []
    all_info: list[tuple[int, object, int]] = []
    for (irrep_idx, (vals, _)) in ed_data.ed_scan_res.items():
        for (e_idx, v) in enumerate(vals):
            all_vals.append(float(v))
            all_info.append((irrep_idx, ed_data.irrep_list[irrep_idx].label, e_idx))
    perm = sorted(range(len(all_vals)), key=lambda k: all_vals[k])

    print("\n--- Lowest 15 eigenvalues ---")
    for i in range(min(15, len(all_vals))):
        ii, label, ei = all_info[perm[i]]
        print(
            f"  E{i + 1} = {all_vals[perm[i]]:.10f}  (k = {label!r}, #{ei + 1})"
        )

    if len(all_vals) >= 4:
        d12 = all_vals[perm[1]] - all_vals[perm[0]]
        d23 = all_vals[perm[2]] - all_vals[perm[1]]
        d34 = all_vals[perm[3]] - all_vals[perm[2]]
        print(f"\n  ΔE₁₂ = {d12:.8f}")
        print(f"  ΔE₂₃ = {d23:.8f}")
        print(f"  ΔE₃₄ = {d34:.8f} (many-body gap)")
        if d12 < 0.15 and d23 < 0.15:
            print(
                "  ✓ Three nearly-degenerate GS detected → FCI signature "
                "(GSD = 3, finite-size split)!"
            )

    # ── Spectrum plot ──
    fig_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
    )
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = ed.plot_spectrum(
        ed_data, shift_to_zero=True,
        title=r"ED Spectrum for Exotic Fermionic Checkerboard FCI "
              r"($t'' = -0.2$) on [3,5] sample at $\nu = 1/3$",
    )
    fig.savefig(os.path.join(fig_dir, "fermion_FCI_checkerboard_spectrum.svg"))

    print("\nDone.")


if __name__ == "__main__":
    main()
