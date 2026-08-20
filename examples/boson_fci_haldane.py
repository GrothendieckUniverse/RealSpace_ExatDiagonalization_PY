#!/usr/bin/env python
"""Example 2: Bosonic fractional Chern insulator on the Haldane honeycomb lattice.

The extended Bose–Hubbard model on the Haldane honeycomb lattice at half
filling of the lower Chern band (ν = 1/2 per band → 3 hard-core bosons on
the 2×3×2 = 12 flattened vertices) hosts the finite-size analogue of the
bosonic Laughlin state (GSD = 2 on the torus)
[D. N. Sheng et al., PRL 107, 146803 (2011)].  Port of
``examples/boson_fci_haldane.jl``.

Usage::

    uv run python examples/boson_fci_haldane.py
"""

from __future__ import annotations

import math
import os

from fractions import Fraction

import realspace_exactdiagonalization_py as ed


def main() -> None:
    sample_size = [2, 3]
    params = dict(ed.params_DNSheng)  # t'' = -0.58 → FCI phase

    model = ed.build_zero_flux_bosonic_fci_second_quantized_model(
        sample_size=sample_size, params=params
    )
    lattice = model.lattice
    n_filled = 3  # half filling of the band → 1/4 of the flattened vertices

    # ── Translation symmetry + ED data ──
    symmetry_group = ed.build_translation_group(lattice)
    ed_data = ed.build_ed_data(
        model, filling_fraction=Fraction(n_filled, lattice.n_site),
        symmetry_group=symmetry_group,
    )

    print(f"\n  Full Hilbert space dim: {math.comb(lattice.n_site, n_filled)}")
    print(
        f"  Orbits: {len(ed_data.orbit_catalog.representative_mask_list)}"
    )
    print(f"  Irreps: {len(ed_data.irrep_list)}  (momenta (k₁,k₂))")

    # ── Symmetry-resolved scan (matrix mode) ──
    ed.ed_scan(ed_data, nev=5, mode="matrix")

    all_vals: list[float] = []
    all_info: list[tuple[int, object, int]] = []
    for (irrep_idx, (vals, _)) in ed_data.ed_scan_res.items():
        for (e_idx, v) in enumerate(vals):
            all_vals.append(float(v))
            all_info.append((irrep_idx, ed_data.irrep_list[irrep_idx].label, e_idx))
    perm = sorted(range(len(all_vals)), key=lambda k: all_vals[k])

    print("\n--- Lowest 6 eigenvalues ---")
    for i in range(min(6, len(all_vals))):
        ii, label, ei = all_info[perm[i]]
        print(
            f"  E{i + 1} = {all_vals[perm[i]]:.10f}  (k = {label!r}, #{ei + 1})"
        )

    # Two nearly-degenerate ground states → bosonic semion FCI on the torus
    d12 = all_vals[perm[1]] - all_vals[perm[0]]
    print(
        f"\n  ΔE₁₂ = {d12:.6f}  "
        "✓ two nearly-degenerate ground states (semion FCI, GSD = 2)"
        if d12 < 0.01
        else f"\n  ΔE₁₂ = {d12:.6f}"
    )
    print(
        f"  GS sectors: {ed_data.irrep_list[all_info[perm[0]][0]].label!r}, "
        f"{ed_data.irrep_list[all_info[perm[1]][0]].label!r}"
    )

    # ── Spectrum plot ──
    fig_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
    )
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = ed.plot_spectrum(ed_data, shift_to_zero=True)
    fig.savefig(os.path.join(fig_dir, "bosonic_FCI_spectrum.svg"))

    print("\nDone.")


if __name__ == "__main__":
    main()
