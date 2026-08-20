#!/usr/bin/env python
"""Example 3: Spin-½ Fermi-Hubbard model on the square lattice.

    H = -t Σ_{⟨i,j⟩,σ} (c†_{iσ} c_{jσ} + h.c.)  +  U Σ_i n_{i↑} n_{i↓}

Spinful fermions handled by flattening the spin degree of freedom into an
interleaved graph: site i↑ → vertex 2i−1, i↓ → vertex 2i.  Lattice: 2×4
(8 spatial sites → 16 graph vertices); half filling per spin → 8 particles;
translation symmetry ℤ₂ × ℤ₄.  Port of
``examples/fermion_hubbard_square.jl``.

Usage::

    uv run python examples/fermion_hubbard_square.py
"""

from __future__ import annotations

import math
import os

import realspace_exactdiagonalization_py as ed


def main() -> None:
    sample_size = [2, 4]  # Lx × Ly
    t = 1.0               # hopping
    U = 8.0               # on-site Hubbard repulsion

    ed_data = ed.build_spinful_hubbard_ed_data(sample_size, t, U)
    n_site = ed_data.second_quantized_model.lattice.n_site
    n_filled = ed_data.n_filled
    print(
        f"[ED] Fermi-Hubbard: {sample_size} → {sample_size[0]*sample_size[1]} "
        f"spatial sites, {n_site} graph vertices"
    )
    print(
        f"[ED]   N_e={n_filled} (N↑=N↓={sample_size[0]*sample_size[1]//2}), "
        f"filling={ed_data.filling_fraction}, t={t}, U={U}"
    )

    print(f"\n  Full Hilbert space dim: {math.comb(n_site, n_filled)}")
    print(
        f"  Orbits: {len(ed_data.orbit_catalog.representative_mask_list)}"
    )
    print(f"  Irreps: {len(ed_data.irrep_list)}  (momenta (k₁,k₂))")

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
        print(
            f"  E{i + 1} = {all_vals[perm[i]]:.10f}  (k = {label!r}, #{ei + 1})"
        )

    # ── Spectrum plot ──
    fig_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
    )
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = ed.plot_spectrum(
        ed_data, shift_to_zero=True,
        title=r"ED Spectrum for Spinful Fermi-Hubbard on [2,4] sample "
              r"at $\nu = 1/2$",
    )
    fig.savefig(os.path.join(fig_dir, "fermion_hubbard_square_spectrum.svg"))

    print("\nDone.")


if __name__ == "__main__":
    main()
