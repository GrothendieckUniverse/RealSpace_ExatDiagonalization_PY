#!/usr/bin/env python
"""Example 1: Spin-½ Heisenberg chain.

    H = J Σ_{⟨i,j⟩} S_i · S_j   (PBC)

via the hard-core boson (Matsubara–Matsuda) mapping
(S^z_i = n_i − ½, S^+_i = b†_i, S^-_i = b_i), absorbed on-site terms, and
half-filling (N_e = N/2, total S^z = 0) with Z_{N_site} translation
symmetry.  Port of ``examples/spin_heisenberg_chain.jl``.

Usage::

    uv run python examples/spin_heisenberg_chain.py
"""

from __future__ import annotations

import math
import os

from fractions import Fraction

import realspace_exactdiagonalization_py as ed


def main() -> None:
    N_SITE = 16  # chain length
    J = 1.0      # exchange coupling (J > 0 = antiferromagnetic)

    # ── 1. Build the ED data (model + Z_N translation symmetry) ──
    ed_data = ed.build_heisenberg_ed_data(N_SITE, J)
    n_site = ed_data.second_quantized_model.lattice.n_site
    n_filled = ed_data.n_filled
    print(
        f"[ED] Heisenberg chain: N_site={n_site}, N_e={n_filled}, "
        f"filling={ed_data.filling_fraction}"
    )

    print(f"\n  Full Hilbert space dim: {math.comb(n_site, n_filled)}")
    print(
        f"  Orbits: {len(ed_data.orbit_catalog.representative_mask_list)}"
    )
    print(f"  Irreps: {len(ed_data.irrep_list)}  (1D momenta k)")

    # ── 2. Run the symmetry-resolved scan ──
    ed.ed_scan(ed_data, nev=5, mode="matrix")

    all_vals: list[float] = []
    all_info: list[tuple[int, object, int]] = []
    for (irrep_idx, (vals, _)) in ed_data.ed_scan_res.items():
        for (e_idx, v) in enumerate(vals):
            all_vals.append(float(v))
            all_info.append((irrep_idx, ed_data.irrep_list[irrep_idx].label, e_idx))
    perm = sorted(range(len(all_vals)), key=lambda k: all_vals[k])

    print("\n--- Lowest 8 eigenvalues ---")
    for i in range(min(8, len(all_vals))):
        ii, label, ei = all_info[perm[i]]
        print(f"  E{i + 1} = {all_vals[perm[i]]:.10f}  (k = {label!r}, #{ei + 1})")

    # known exact result: E0/N = 1/4 - ln(2) ≈ -0.44314718 per site
    print(
        f"\n  E0/N = {min(all_vals) / N_SITE:.10f}  "
        "(Bethe ansatz: 1/4 − ln 2 ≈ −0.44314718)"
    )

    # ── 3. Spectrum plot ──
    ed.print_spectrum(ed_data, shift_to_zero=True)
    fig, ax = ed.plot_spectrum(ed_data, shift_to_zero=True)
    fig_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
    )
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "heisenberg_chain_spectrum.svg"))

    print("\nDone.")


if __name__ == "__main__":
    main()
