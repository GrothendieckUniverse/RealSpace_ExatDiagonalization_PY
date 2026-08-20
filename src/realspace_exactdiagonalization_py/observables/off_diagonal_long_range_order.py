"""One-body density matrix in k-space (off-diagonal long-range order).

Faithful port of ``observables/off_diagonal_long_range_order.jl`` from the
Julia package ``RealSpace_ExactDiagonalization.jl``.

.. math::
   \\rho_{ij} &= \\langle a_i^\\dagger a_j\\rangle \\\\
   \\rho(k)   &= \\frac{1}{N}\\sum_{i,j} e^{ik\\cdot(r_i-r_j)}
                \\langle a_i^\\dagger a_j\\rangle

This is the primary diagnostic for superfluidity / Bose condensation in
translation-invariant PBC systems.  The one-body density matrix
:math:`\\rho[i,j] = \\langle a_i^\\dagger a_j\\rangle` is computed by an exact
orbit-basis contraction (no full-Fock-space expansion): for each projected-basis
representative we loop over its symmetry orbit, apply the local one-body
operator, and project the scattered mask back to the same sector.
"""

from __future__ import annotations

import os
from fractions import Fraction

import numpy as np

from ..utils.bitwise_operations import (
    empty_site_for_mask,
    is_site_occupied,
    occupy_site_for_mask,
)
from ..second_quantized_model import Real_Space_Second_Quantized_Model
from ..symmetry_resolved_ed import (
    CanonicalMap,
    Symmetry_Resolved_ED_Data,
    apply_operation_to_mask,
    group_order,
    hopping_phase_for_stats,
    project_to_sector,
)
from .static_structure_factor import _bootstrap_ed_sector, _draw_bz_boundary, _precompute_phases


def off_diagonal_long_range_order(
    model: Real_Space_Second_Quantized_Model,
    sector_label,
    *,
    target_eigval_idx: int = 1,
    filling_fraction: Fraction,
    ed_mode: str = "matrix",
    ed_data: Symmetry_Resolved_ED_Data | None = None,
) -> np.ndarray:
    """One-body density matrix :math:`\\rho[i,j] = \\langle \\psi| a_i^\\dagger a_j |\\psi\\rangle`.

    Julia counterpart: ``off_diagonal_long_range_order``.

    Args:
        model: the second-quantized model.
        sector_label: ``"identity"`` (full real-space ED) or a momentum tuple
            ``(k1, k2)`` for translation-resolved ED.
        target_eigval_idx: **1-based** target eigenstate index.
        filling_fraction: particles per flattened vertex.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.

    Returns:
        np.ndarray: complex ``(n_site, n_site)`` matrix with
        ``odlro[i, j] = ⟨a_{i+1}^† a_{j+1}⟩`` (0-based; the Julia counterpart
        returns a 1-based ``Matrix{ComplexF64}``).
    """
    lattice = model.lattice
    n_site = lattice.n_site
    stats = model.particle_statistics

    ed_data, basis, c = _bootstrap_ed_sector(
        model, sector_label, filling_fraction, target_eigval_idx, ed_mode, ed_data
    )

    odlro = np.zeros((n_site, n_site), dtype=np.complex128)

    G = basis.symmetry_group
    nG = group_order(G)
    inv_nG = 1.0 / nG
    inv_sqrt_nG = 1.0 / np.sqrt(nG)
    cmap = CanonicalMap(ed_data.symmetry_group, stats, ed_data.orbit_catalog)

    for col, repr_mask in enumerate(basis.representative_mask_list):
        c_col = c[col]
        if c_col == 0:
            continue

        # In the normalized projected basis |[s];χ⟩ = 1/√(|G||Stab(s)|) Σ_g
        # χ(g)^* U_g |s⟩.  For diagonal n_j, stabilizer-related copies of the
        # same raw Fock state add coherently, giving the orbit-average weight
        # |c_col|²/|G| for every group image of the representative.
        diag_weight = abs(c_col) ** 2 * inv_nG
        ket_norm = inv_sqrt_nG / np.sqrt(basis.stabilizer_order_list[col])

        for gidx, op in enumerate(G.operations):
            ket_mask, ket_group_phase = apply_operation_to_mask(
                repr_mask, op, stats
            )
            ket_basis_amp = ket_norm * np.conj(basis.irrep.values[gidx]) * ket_group_phase
            ket_amp = c_col * ket_basis_amp

            tmp_j = ket_mask
            while tmp_j != 0:
                lsb_j = tmp_j & -tmp_j
                j = (lsb_j.bit_length() - 1) + 1
                tmp_j ^= lsb_j

                odlro[j - 1, j - 1] += diag_weight

                for i in range(1, n_site + 1):
                    if i == j:
                        continue
                    if is_site_occupied(ket_mask, i):
                        continue

                    new_mask = empty_site_for_mask(ket_mask, j)
                    new_mask = occupy_site_for_mask(new_mask, i)

                    proj = project_to_sector(new_mask, basis, cmap)
                    if proj is None:
                        continue
                    row, proj_coeff = proj

                    # If project_to_sector(new_mask) = (row, p), the
                    # coefficient of |new_mask⟩ in the normalized row basis is
                    # √(|Stab(row)|/|G|) · conj(p).
                    bra_basis_amp = (
                        np.sqrt(basis.stabilizer_order_list[row])
                        * inv_sqrt_nG
                        * np.conj(proj_coeff)
                    )
                    bra_amp = c[row] * bra_basis_amp

                    phase = hopping_phase_for_stats(stats, ket_mask, j, i)
                    odlro[i - 1, j - 1] += np.conj(bra_amp) * ket_amp * phase

    return odlro


def compute_odlro_map(
    model: Real_Space_Second_Quantized_Model,
    sector_label,
    *,
    target_eigval_idx: int = 1,
    filling_fraction: Fraction,
    k_resolution: int = 61,
    ed_mode: str = "matrix",
    ed_data: Symmetry_Resolved_ED_Data | None = None,
):
    """Momentum-space ODLRO :math:`\\rho(k)` on a dense grid over ``[-1.5π, 1.5π]²``.

    Julia counterpart: ``compute_odlro_map``.

    Args:
        model: the second-quantized model.
        sector_label: ``"identity"`` or a momentum tuple.
        target_eigval_idx: **1-based** target eigenstate index.
        filling_fraction: particles per flattened vertex.
        k_resolution: number of grid points per direction.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.

    Returns:
        tuple: ``(kx, ky, odlro_map)`` with ``odlro_map[i, j] = ρ(kx[i], ky[j])``
        (column-major, matching the Julia ``reshape`` layout).
    """
    lattice = model.lattice
    n_site = lattice.n_site

    odlro = off_diagonal_long_range_order(
        model,
        sector_label,
        target_eigval_idx=target_eigval_idx,
        filling_fraction=filling_fraction,
        ed_mode=ed_mode,
        ed_data=ed_data,
    )

    span = 3 * np.pi
    kx = np.linspace(-span / 2, span / 2, num=k_resolution)
    ky = np.linspace(-span / 2, span / 2, num=k_resolution)
    q_points = [[x, y] for y in ky for x in kx]

    phases = _precompute_phases(q_points, lattice.site_cart_list)

    n_q = len(q_points)
    odlro_flat = np.zeros(n_q, dtype=np.float64)
    for q_idx in range(n_q):
        ph = phases[q_idx, :]
        acc = np.dot(np.dot(np.conj(ph), odlro), ph)
        odlro_flat[q_idx] = float(np.real(acc)) / n_site

    odlro_map = odlro_flat.reshape(k_resolution, k_resolution, order="F")
    return kx, ky, odlro_map


def plot_odlro_map(
    model: Real_Space_Second_Quantized_Model,
    sector_label,
    *,
    target_eigval_idx: int = 1,
    filling_fraction: Fraction,
    k_resolution: int = 61,
    ed_mode: str = "matrix",
    ed_data: Symmetry_Resolved_ED_Data | None = None,
    fig_path: str | None = None,
    title: str = "Momentum distribution ρ(k)",
):
    """Compute and plot the dense momentum-space ODLRO map with the BZ boundary.

    Julia counterpart: ``plot_odlro_map`` (CairoMakie → matplotlib).

    Args:
        model: the second-quantized model.
        sector_label: ``"identity"`` or a momentum tuple.
        target_eigval_idx: **1-based** target eigenstate index.
        filling_fraction: particles per flattened vertex.
        k_resolution: number of grid points per direction.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.
        fig_path: optional path to save the figure.
        title: figure title.

    Returns:
        tuple: ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    kx, ky, odlro_map = compute_odlro_map(
        model,
        sector_label,
        target_eigval_idx=target_eigval_idx,
        filling_fraction=filling_fraction,
        k_resolution=k_resolution,
        ed_mode=ed_mode,
        ed_data=ed_data,
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax.set_xlabel("k_x")
    ax.set_ylabel("k_y")
    ax.set_title(title)
    ax.set_aspect("equal")
    hm = ax.pcolormesh(kx, ky, odlro_map.T, shading="auto", cmap="viridis")
    _draw_bz_boundary(ax, model.lattice)
    fig.colorbar(hm, ax=ax, label="ρ(k)")

    if fig_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
        fig.savefig(fig_path)
    return fig, ax


def plot_odlro_map_panels(
    maps,
    *,
    fig_path: str | None = None,
    title: str = "Momentum distribution ρ(k)",
):
    """Plot several precomputed ODLRO maps in one row with a unified scale.

    Julia counterpart: ``plot_odlro_map_panels``.

    Each entry of ``maps`` exposes fields ``kx``, ``ky``, ``values``,
    ``lattice`` and ``title``.

    Args:
        maps: list of map entries.
        fig_path: optional path to save the figure.
        title: overall figure title.

    Returns:
        tuple: ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt

    if not maps:
        raise ValueError("maps must not be empty.")

    vmin = min(np.min(m.values) for m in maps)
    vmax = max(np.max(m.values) for m in maps)

    fig, axes = plt.subplots(1, len(maps), figsize=(4.2 * len(maps) + 1.1, 4.2))
    if len(maps) == 1:
        axes = [axes]
    hm_ref = None
    for idx, m in enumerate(maps):
        ax = axes[idx]
        ax.set_xlabel("k_x")
        ax.set_ylabel("k_y")
        ax.set_title(m.title)
        ax.set_aspect("equal")
        hm = ax.pcolormesh(
            m.kx, m.ky, m.values.T, shading="auto", cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        _draw_bz_boundary(ax, m.lattice)
        if hm_ref is None:
            hm_ref = hm
    fig.suptitle(title, fontsize=18)
    fig.colorbar(hm_ref, ax=axes, label="ρ(k)")

    if fig_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
        fig.savefig(fig_path)
    return fig, axes


__all__ = [
    "off_diagonal_long_range_order",
    "compute_odlro_map",
    "plot_odlro_map",
    "plot_odlro_map_panels",
]
