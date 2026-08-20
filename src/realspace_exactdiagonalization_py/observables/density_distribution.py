"""Real-space occupation distribution of graph vertices.

Faithful port of ``observables/density_distribution.jl`` from the Julia
package ``RealSpace_ExactDiagonalization.jl``.

After extensive analysis the Julia module was simplified: the
symmetry-resolved single-sector density operator is mathematically guaranteed
to give uniform :math:`\\langle n_i\\rangle` for any translation-invariant
Hamiltonian with periodic boundary conditions, regardless of the underlying
phase.  This module therefore retains only the full real-space ED path
(identity symmetry group), which requires **open boundary conditions** — the
only setting where a single ground-state eigenvector can display genuine
real-space density modulation.

For diagnosing charge order / superfluidity in PBC systems use
:mod:`realspace_exactdiagonalization_py.observables.static_structure_factor`
(connected :math:`S(q)`) and
:mod:`realspace_exactdiagonalization_py.observables.off_diagonal_long_range_order`
(:math:`\\rho(k) = \\langle a_i^\\dagger a_j\\rangle`).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable

import numpy as np

from ..second_quantized_model import (
    Real_Space_Second_Quantized_Model,
    update_second_quantized_model_with_twisted_phases,
)
from ..symmetry_resolved_ed import (
    Symmetry_Resolved_ED_Data,
    build_ed_data,
    build_identity_group,
    build_symmetry_sector_basis,
    ed_scan,
)


def vertices_occupation_distribution_full_ed(
    model: Real_Space_Second_Quantized_Model,
    *,
    target_eigval_idx: int = 1,
    filling_fraction: Fraction,
    flavor_filter: Callable[[int], bool] | None = None,
    ed_mode: str = "matrix",
    ed_data: Symmetry_Resolved_ED_Data | None = None,
) -> np.ndarray:
    """Compute :math:`\\langle n_i\\rangle` from a full real-space ED eigenvector.

    Julia counterpart: ``vertices_occupation_distribution_full_ed``.  Uses the
    identity symmetry group (each symmetry orbit = one raw Fock mask) and
    therefore evaluates the occupation from the raw full-Hilbert-space
    eigenvector.

    .. warning::
        This function requires **open boundary conditions**.  For a
        translation-invariant Hamiltonian with periodic boundary conditions a
        single symmetry-sector ground state gives identically uniform density
        (a mathematical consequence of :math:`[H,T]=0` and the non-degeneracy
        of the sector).  The function asserts ``not any(pbc_indicator)``.

    Args:
        model: the second-quantized model.
        target_eigval_idx: **1-based** index of the target eigenvector.
        filling_fraction: particles per flattened vertex (exact rational).
        flavor_filter: optional boolean site filter :math:`i\\mapsto` bool
            selecting the flavour component (e.g. a sublattice).  ``None``
            selects all sites.
        ed_mode: ED mode (``"matrix"`` or ``"matrixfree"``).
        ed_data: optional precomputed :class:`Symmetry_Resolved_ED_Data`; if
            given, the bootstrap scan is skipped.

    Returns:
        np.ndarray: real array of length ``n_site`` with ``density[i] ==
        ⟨n_{i+1}⟩`` (0-based; the Julia counterpart returns a 1-based
        ``Vector{Float64}``).
    """
    if any(model.lattice.pbc_indicator):
        raise AssertionError(
            "vertices_occupation_distribution_full_ed requires open boundary "
            "conditions.  For PBC systems, use `static_structure_factor` or "
            "`off_diagonal_long_range_order` instead."
        )

    if flavor_filter is None:
        flavor_filter = lambda i: True  # noqa: E731

    n_site = model.lattice.n_site

    if ed_data is None:
        flux0 = [0.0] * model.lattice.dim
        update_second_quantized_model_with_twisted_phases(
            model, twisted_phases_over_2π=flux0
        )
        G = build_identity_group(n_site)
        ed_data = build_ed_data(
            model, filling_fraction=filling_fraction, symmetry_group=G
        )
        ed_scan(ed_data, nev=max(target_eigval_idx, 2), mode=ed_mode)

    irrep_idx = None
    for idx, irrep in enumerate(ed_data.irrep_list):
        if irrep.label == "identity":
            irrep_idx = idx
            break
    if irrep_idx is None:
        raise ValueError(
            "Full real-space ED sector :identity was not found in ed_data."
        )
    if irrep_idx not in ed_data.ed_scan_res:
        raise ValueError("Full real-space ED sector :identity was not scanned.")

    _, vecs = ed_data.ed_scan_res[irrep_idx]
    if target_eigval_idx > vecs.shape[1]:
        raise ValueError(
            f"target_eigval_idx={target_eigval_idx} exceeds the "
            f"{vecs.shape[1]} computed full-ED eigenvectors."
        )

    basis = build_symmetry_sector_basis(
        ed_data.orbit_catalog, ed_data.irrep_list[irrep_idx]
    )
    c = vecs[:, target_eigval_idx - 1]
    if len(c) != len(basis.representative_mask_list):
        raise ValueError(
            "Full-ED eigenvector length does not match identity-basis dimension."
        )

    density = np.zeros(n_site, dtype=np.float64)
    for col, mask in enumerate(basis.representative_mask_list):
        w = abs(c[col]) ** 2
        if w == 0.0:
            continue

        tmp = mask
        while tmp != 0:
            lsb = tmp & -tmp
            vertex = (lsb.bit_length() - 1) + 1
            if flavor_filter(vertex):
                density[vertex - 1] += w
            tmp ^= lsb

    return density


__all__ = ["vertices_occupation_distribution_full_ed"]
