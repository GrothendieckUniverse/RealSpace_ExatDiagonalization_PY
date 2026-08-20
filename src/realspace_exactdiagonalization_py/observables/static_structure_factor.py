"""Connected density-density correlation in k-space (static structure factor).

Faithful port of ``observables/static_structure_factor.jl`` from the Julia
package ``RealSpace_ExactDiagonalization.jl``.

.. math::
   S^{\\alpha\\beta}(q) = \\frac{1}{N}\\sum_{i,j}
   e^{iq\\cdot(r_i-r_j)}
   \\big(\\langle n_i^\\alpha n_j^\\beta\\rangle
        - \\langle n_i^\\alpha\\rangle\\langle n_j^\\beta\\rangle\\big)

This is the primary diagnostic for charge order in translation-invariant PBC
systems: a CDW/Wigner crystal shows sharp Bragg peaks at the ordering
wavevector :math:`Q`; an FCI or superfluid shows a smooth, featureless
:math:`S(q)`.  Both the diagonal density :math:`\\hat n_i` and the off-diagonal
one-body term are computed **directly in the symmetry-sector (orbit) basis** to
avoid expanding the eigenvector into the full Fock space.
"""

from __future__ import annotations

import cmath
import math
import os
from fractions import Fraction
from types import SimpleNamespace
from typing import Callable

import numpy as np

from ..second_quantized_model import (
    Particle_Statistics,
    Real_Space_Second_Quantized_Model,
    update_second_quantized_model_with_twisted_phases,
)
from ..symmetry_resolved_ed import (
    Symmetry_Resolved_ED_Data,
    Symmetry_Sector_Basis,
    apply_operation_to_mask,
    build_ed_data,
    build_identity_group,
    build_symmetry_sector_basis,
    build_translation_group,
    ed_scan,
    group_order,
)


# ═══════════════════════════════════════════════════════════════════════════
# Internal: connected density-density correlation matrix C[i, j]
# ═══════════════════════════════════════════════════════════════════════════


def _build_density_correlation_matrix(
    C: np.ndarray,
    density_a: np.ndarray,
    density_b: np.ndarray,
    c: np.ndarray,
    basis: Symmetry_Sector_Basis,
    n_site: int,
    flavor_a: Callable[[int], bool],
    flavor_b: Callable[[int], bool],
    particle_statistics: Particle_Statistics,
    *,
    subtract_disconnected: bool = True,
) -> np.ndarray:
    """Build the connected density-density correlation matrix (in place).

    Julia counterpart: ``_build_density_correlation_matrix!``.  For each
    representative mask ``r`` with amplitude :math:`|c_{col}|^2`, sum over all
    :math:`|G|` orbit members :math:`\\pi_g(r)`:

    .. math::
        \\langle n_i n_j\\rangle = \\sum_{col} |c_{col}|^2\\,
        \\frac{1}{|G|}\\sum_g n_i(\\pi_g(r_{col}))\\,n_j(\\pi_g(r_{col})).

    Args:
        C: ``(n_site, n_site)`` real buffer, zeroed and overwritten.
        density_a: length-``n_site`` real buffer for :math:`\\langle n^a_i\\rangle`.
        density_b: length-``n_site`` real buffer for :math:`\\langle n^b_i\\rangle`.
        c: sector eigenvector amplitudes.
        basis: the symmetry-sector basis.
        n_site: number of vertices.
        flavor_a: boolean site filter for flavour a.
        flavor_b: boolean site filter for flavour b.
        particle_statistics: particle statistics of the model.
        subtract_disconnected: subtract :math:`\\langle n_i\\rangle\\langle n_j\\rangle`.

    Returns:
        np.ndarray: the same (mutated) ``C`` buffer.
    """
    C.fill(0.0)
    density_a.fill(0.0)
    density_b.fill(0.0)

    G = basis.symmetry_group
    nG = group_order(G)
    nG_inv = 1.0 / nG

    for col, repr_mask in enumerate(basis.representative_mask_list):
        w = abs(c[col]) ** 2
        if w == 0.0:
            continue
        w_eff = w * nG_inv

        for op in G.operations:
            shifted, _ = apply_operation_to_mask(
                repr_mask, op, particle_statistics
            )

            # Collect occupied vertices of this orbit member.
            tmp = shifted
            while tmp != 0:
                lsb = tmp & -tmp
                i = (lsb.bit_length() - 1) + 1
                if flavor_a(i):
                    density_a[i - 1] += w_eff
                if flavor_b(i):
                    density_b[i - 1] += w_eff
                tmp ^= lsb

            # Two-point contribution: n_i n_j for i, j both occupied.
            tmp2 = shifted
            while tmp2 != 0:
                lsb2 = tmp2 & -tmp2
                i = (lsb2.bit_length() - 1) + 1
                tmp2 ^= lsb2
                if not flavor_a(i):
                    continue
                tmp3 = shifted
                while tmp3 != 0:
                    lsb3 = tmp3 & -tmp3
                    j = (lsb3.bit_length() - 1) + 1
                    tmp3 ^= lsb3
                    if flavor_b(j):
                        C[i - 1, j - 1] += w_eff

    # Subtract disconnected part: ⟨n_i n_j⟩_c = ⟨n_i n_j⟩ − ⟨n_i⟩⟨n_j⟩.
    if subtract_disconnected:
        for j in range(n_site):
            for i in range(n_site):
                C[i, j] -= density_a[i] * density_b[j]

    return C


# ═══════════════════════════════════════════════════════════════════════════
# Fourier-transform helpers
# ═══════════════════════════════════════════════════════════════════════════


def _precompute_phases(q_points, positions) -> np.ndarray:
    """Precompute phases :math:`e^{-i q\\cdot r_i}` for all ``(q, site)`` pairs.

    Julia counterpart: the two ``_precompute_phases`` methods (one taking a
    :class:`Uniform_Grids` and one taking an explicit ``q_points`` list).  The
    Python port keeps a single function parameterized by the ``q_points`` list;
    callers pass ``kgrid.site_cart_list`` for the lattice-dispatched case.

    Args:
        q_points: list of momentum vectors ``q`` (each a length-``dim`` sequence).
        positions: list of site Cartesian positions (length-``dim`` sequences).

    Returns:
        np.ndarray: complex ``(n_q, n_site)`` array with
        ``phases[q_idx, i] = exp(-i q·positions[i])``.
    """
    n_q = len(q_points)
    n_site = len(positions)
    pos = np.asarray(
        [np.asarray(p, dtype=np.float64) for p in positions]
    ).reshape(n_site, -1)
    phases = np.empty((n_q, n_site), dtype=np.complex128)
    for q_idx, q in enumerate(q_points):
        qv = np.asarray(q, dtype=np.float64)
        phases[q_idx, :] = np.exp(-1j * (pos @ qv))
    return phases


def _fourier_transform(
    S_q: np.ndarray, phases: np.ndarray, C: np.ndarray, n_site: int
) -> np.ndarray:
    """Fourier-transform ``C[i, j]`` to ``S(q)`` at all q (in place).

    Julia counterpart: ``_fourier_transform!``.

    Args:
        S_q: length-``n_q`` real buffer to be overwritten.
        phases: complex ``(n_q, n_site)`` phase array from
            :func:`_precompute_phases`.
        C: real ``(n_site, n_site)`` correlation matrix.
        n_site: number of vertices.

    Returns:
        np.ndarray: the same (mutated) ``S_q`` buffer.
    """
    n_q = phases.shape[0]
    for q_idx in range(n_q):
        ph = phases[q_idx, :]
        acc = float(np.real(np.dot(np.dot(ph, C), np.conj(ph))))
        S_q[q_idx] = acc / n_site
    return S_q


# ═══════════════════════════════════════════════════════════════════════════
# Brillouin-zone boundary helpers for dense k-space visualisations
# ═══════════════════════════════════════════════════════════════════════════


def _reciprocal_basis_vectors(lattice):
    """Reciprocal basis vectors :math:`b_1, b_2` of a 2-D lattice.

    Julia counterpart: ``_reciprocal_basis_vectors``.

    Args:
        lattice: the real-space lattice.

    Returns:
        list[np.ndarray]: ``[b1, b2]`` such that
        :math:`a_i\\cdot b_j = 2\\pi\\,\\delta_{ij}`.
    """
    if lattice.dim != 2:
        raise ValueError("BZ boundary plotting is implemented only for 2D lattices.")
    a1 = np.asarray(lattice.brav_vec_list[0], dtype=np.float64)
    a2 = np.asarray(lattice.brav_vec_list[1], dtype=np.float64)
    A = np.array([[a1[0], a2[0]], [a1[1], a2[1]]])
    B = 2 * np.pi * np.linalg.inv(A).T
    return [B[:, 0], B[:, 1]]


def _clip_polygon_by_halfplane(poly, normal, offset, *, atol: float = 1e-12):
    """Sutherland–Hodgman clip of a polygon by the half-plane n·x ≤ offset.

    Julia counterpart: ``_clip_polygon_by_halfplane``.

    Args:
        poly: list of 2-D points (each a length-2 sequence).
        normal: outward half-plane normal.
        offset: half-plane offset.
        atol: tolerance for the point-in-half-plane test.

    Returns:
        list[np.ndarray]: the clipped polygon vertex list.
    """
    if not poly:
        return poly
    clipped: list[np.ndarray] = []
    n = len(poly)
    normal = np.asarray(normal, dtype=np.float64)
    for idx in range(n):
        p = np.asarray(poly[idx], dtype=np.float64)
        q = np.asarray(poly[(idx + 1) % n], dtype=np.float64)
        fp = float(np.dot(normal, p) - offset)
        fq = float(np.dot(normal, q) - offset)
        p_in = fp <= atol
        q_in = fq <= atol

        if p_in and q_in:
            clipped.append(q)
        elif p_in and not q_in:
            t = fp / (fp - fq)
            clipped.append(p + t * (q - p))
        elif (not p_in) and q_in:
            t = fp / (fp - fq)
            clipped.append(p + t * (q - p))
            clipped.append(q)
    return clipped


def _first_bz_polygon(lattice):
    """First Brillouin-zone polygon (Wigner–Seitz cell of the reciprocal lattice).

    Julia counterpart: ``_first_bz_polygon``.  A large square is clipped by the
    bisecting half-planes of the smallest reciprocal lattice vectors.

    Args:
        lattice: the real-space lattice.

    Returns:
        list[np.ndarray]: the ordered first-BZ boundary polygon vertices.
    """
    b1, b2 = _reciprocal_basis_vectors(lattice)
    extent = 2.5 * max(float(np.linalg.norm(b1)), float(np.linalg.norm(b2)))
    poly = [
        np.array([-extent, -extent]),
        np.array([extent, -extent]),
        np.array([extent, extent]),
        np.array([-extent, extent]),
    ]

    Gs: list[np.ndarray] = []
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            if n1 == 0 and n2 == 0:
                continue
            Gs.append(n1 * b1 + n2 * b2)
    Gs.sort(key=lambda g: float(np.linalg.norm(g)))

    for G in Gs:
        poly = _clip_polygon_by_halfplane(poly, G, float(np.dot(G, G)) / 2)
        if not poly:
            break
    return poly


def _draw_bz_boundary(ax, lattice):
    """Overlay the first Brillouin-zone boundary on a matplotlib axis.

    Julia counterpart: ``_draw_bz_boundary!`` (Makie ``lines!``).

    Args:
        ax: the matplotlib axis.
        lattice: the real-space lattice.

    Returns:
        object: the same ``ax``.
    """
    poly = _first_bz_polygon(lattice)
    if not poly:
        return ax
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    xs.append(xs[0])
    ys.append(ys[0])
    ax.plot(xs, ys, color="black", linewidth=2.5)
    return ax


def _fold_momentum_to_first_bz(q, lattice):
    """Fold a momentum vector into the first Brillouin zone.

    Julia counterpart: ``_fold_momentum_to_first_bz``.  Among the 25 shifts
    :math:`q - n_1 b_1 - n_2 b_2` with :math:`n_1, n_2 \\in \\{-2,\\dots,2\\}`,
    the one with the smallest squared norm is returned.

    Args:
        q: length-``dim`` momentum vector.
        lattice: the real-space lattice.

    Returns:
        np.ndarray: the folded momentum vector (a copy of ``q`` for dim ≠ 2).
    """
    if lattice.dim != 2:
        return np.asarray(q, dtype=np.float64).copy()
    b1, b2 = _reciprocal_basis_vectors(lattice)
    q = np.asarray(q, dtype=np.float64)
    best = q.copy()
    best_norm = float(np.dot(best, best))

    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            folded = q - n1 * b1 - n2 * b2
            folded_norm = float(np.dot(folded, folded))
            if folded_norm < best_norm:
                best = folded
                best_norm = folded_norm
    return best


def _fold_momenta_to_first_bz(q_points, lattice):
    """Fold a list of momentum vectors into the first Brillouin zone.

    Julia counterpart: ``_fold_momenta_to_first_bz``.

    Args:
        q_points: list of momentum vectors.
        lattice: the real-space lattice.

    Returns:
        list[np.ndarray]: the folded momentum vectors.
    """
    return [_fold_momentum_to_first_bz(np.asarray(q, dtype=np.float64), lattice) for q in q_points]


# ═══════════════════════════════════════════════════════════════════════════
# Shared ED bootstrap
# ═══════════════════════════════════════════════════════════════════════════


def _bootstrap_ed_sector(
    model: Real_Space_Second_Quantized_Model,
    sector_label,
    filling_fraction: Fraction,
    target_eigval_idx: int,
    ed_mode: str,
    ed_data: Symmetry_Resolved_ED_Data | None,
):
    """Bootstrap ED for a sector and return ``(ed_data, basis, c)``.

    Julia counterpart: ``_bootstrap_ed_sector``.  If ``ed_data`` is ``None``,
    the model is set to zero twisted phases, the identity (for
    ``sector_label == "identity"``) or translation group is built, the ED data
    is constructed and scanned; then the target sector eigenvector is resolved.

    Args:
        model: the second-quantized model.
        sector_label: ``"identity"`` or a momentum tuple ``(k1, k2)``.
        filling_fraction: particles per flattened vertex.
        target_eigval_idx: **1-based** target eigenstate index.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.

    Returns:
        tuple: ``(ed_data, basis, c)`` where ``c`` is the target sector
        eigenvector.
    """
    lattice = model.lattice
    n_site = lattice.n_site

    if ed_data is None:
        flux0 = [0.0] * lattice.dim
        update_second_quantized_model_with_twisted_phases(
            model, twisted_phases_over_2π=flux0
        )

        G = (
            build_identity_group(n_site)
            if sector_label == "identity"
            else build_translation_group(lattice, flux0)
        )
        ed_data = build_ed_data(
            model, filling_fraction=filling_fraction, symmetry_group=G
        )

        scanned = None if sector_label == "identity" else [sector_label]
        ed_scan(
            ed_data,
            nev=max(target_eigval_idx, 2),
            mode=ed_mode,
            scanned_sectors=scanned,
        )

    irrep_idx = None
    for idx, irrep in enumerate(ed_data.irrep_list):
        if irrep.label == sector_label:
            irrep_idx = idx
            break
    if irrep_idx is None:
        raise ValueError(f"Sector {sector_label!r} not found in ed_data.irrep_list.")
    if irrep_idx not in ed_data.ed_scan_res:
        raise ValueError(f"Sector {sector_label!r} was not scanned.")

    _, vecs = ed_data.ed_scan_res[irrep_idx]
    if target_eigval_idx > vecs.shape[1]:
        raise ValueError(
            f"target_eigval_idx={target_eigval_idx} exceeds {vecs.shape[1]} eigenvectors."
        )

    basis = build_symmetry_sector_basis(
        ed_data.orbit_catalog, ed_data.irrep_list[irrep_idx]
    )
    c = vecs[:, target_eigval_idx - 1]

    return ed_data, basis, c


# ═══════════════════════════════════════════════════════════════════════════
# Public API — static structure factor
# ═══════════════════════════════════════════════════════════════════════════


def static_structure_factor(
    model: Real_Space_Second_Quantized_Model,
    sector_label,
    *,
    target_eigval_idx: int = 1,
    filling_fraction: Fraction,
    flavor_a: Callable[[int], bool] | None = None,
    flavor_b: Callable[[int], bool] | None = None,
    ed_mode: str = "matrix",
    ed_data: Symmetry_Resolved_ED_Data | None = None,
):
    """Connected static structure factor on the finite-torus momentum mesh.

    Julia counterpart: ``static_structure_factor``.

    .. math::
        S^{\\alpha\\beta}(q) = \\frac{1}{N}\\sum_{i,j}
        e^{iq\\cdot(r_i-r_j)}
        \\big(\\langle n_i^\\alpha n_j^\\beta\\rangle
             - \\langle n_i^\\alpha\\rangle\\langle n_j^\\beta\\rangle\\big)

    The q-point mesh is the reciprocal grid built from the lattice via
    ``tightbinding_py.initialize_uniform_grids_from_lattice``.

    Args:
        model: the second-quantized model.
        sector_label: ``"identity"`` (full real-space ED) or a momentum tuple.
        target_eigval_idx: **1-based** target eigenstate index.
        filling_fraction: particles per flattened vertex.
        flavor_a: boolean site filter for flavour a (``None`` → all sites).
        flavor_b: boolean site filter for flavour b (``None`` → all sites).
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.

    Returns:
        tuple: ``(q_points, S_q)`` — the list of q-vectors and the real
        ``S(q)`` array.
    """
    lattice = model.lattice
    n_site = lattice.n_site
    stats = model.particle_statistics

    if flavor_a is None:
        flavor_a = lambda i: True  # noqa: E731
    if flavor_b is None:
        flavor_b = lambda i: True  # noqa: E731

    ed_data, basis, c = _bootstrap_ed_sector(
        model, sector_label, filling_fraction, target_eigval_idx, ed_mode, ed_data
    )

    C = np.zeros((n_site, n_site), dtype=np.float64)
    density_a = np.zeros(n_site, dtype=np.float64)
    density_b = np.zeros(n_site, dtype=np.float64)
    _build_density_correlation_matrix(
        C, density_a, density_b, c, basis, n_site, flavor_a, flavor_b, stats
    )

    from tightbinding_py import initialize_uniform_grids_from_lattice

    kgrid = initialize_uniform_grids_from_lattice(lattice)
    phases = _precompute_phases(kgrid.site_cart_list, lattice.site_cart_list)
    S_q = np.zeros(kgrid.nsite, dtype=np.float64)
    _fourier_transform(S_q, phases, C, n_site)

    return list(kgrid.site_cart_list), S_q


def _manifold_state_components(state):
    """Normalize a manifold state to a ``(sector, level)`` tuple.

    Julia counterpart: ``_manifold_state_components``.

    Args:
        state: an object exposing ``sector``/``level`` attributes, a
            ``{"sector": ..., "level": ...}`` dict, or a length-2 tuple.

    Returns:
        tuple: ``(sector_tuple, level_int)``.
    """
    if hasattr(state, "sector") and hasattr(state, "level"):
        return tuple(int(x) for x in state.sector), int(state.level)
    if isinstance(state, dict) and "sector" in state and "level" in state:
        return tuple(int(x) for x in state["sector"]), int(state["level"])
    if isinstance(state, tuple) and len(state) == 2:
        return tuple(int(x) for x in state[0]), int(state[1])
    raise ValueError(f"A manifold state must provide `(sector, level)`; got {state!r}.")


def _manifold_connected_density_correlation(
    model: Real_Space_Second_Quantized_Model,
    manifold_states,
    *,
    weights=None,
    filling_fraction: Fraction,
    flavor_a: Callable[[int], bool] | None = None,
    flavor_b: Callable[[int], bool] | None = None,
    ed_mode: str = "matrix",
    ed_data: Symmetry_Resolved_ED_Data | None = None,
) -> np.ndarray:
    """Connected density correlator of an incoherent manifold projector.

    Julia counterpart: ``_manifold_connected_density_correlation``.

    Args:
        model: the second-quantized model.
        manifold_states: list of ``(sector, level)`` manifold states.
        weights: optional real weights (default: uniform).
        filling_fraction: particles per flattened vertex.
        flavor_a: boolean site filter for flavour a.
        flavor_b: boolean site filter for flavour b.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.

    Returns:
        np.ndarray: the ``(n_site, n_site)`` connected correlation matrix.
    """
    if not manifold_states:
        raise ValueError("The manifold-state list must not be empty.")

    if weights is None:
        normalized_weights = [1.0 / len(manifold_states)] * len(manifold_states)
    else:
        if len(weights) != len(manifold_states):
            raise ValueError(
                f"Received {len(weights)} weights for {len(manifold_states)} states."
            )
        if any(w < 0 for w in weights):
            raise ValueError("Manifold weights must be nonnegative.")
        total = sum(weights)
        if total <= 0:
            raise ValueError("At least one manifold weight must be positive.")
        normalized_weights = [float(w) / total for w in weights]

    if flavor_a is None:
        flavor_a = lambda i: True  # noqa: E731
    if flavor_b is None:
        flavor_b = lambda i: True  # noqa: E731

    n_site = model.lattice.n_site
    C = np.zeros((n_site, n_site), dtype=np.float64)
    density_a = np.zeros(n_site, dtype=np.float64)
    density_b = np.zeros(n_site, dtype=np.float64)

    for state, weight in zip(manifold_states, normalized_weights):
        sector, level = _manifold_state_components(state)
        _, basis, vector = _bootstrap_ed_sector(
            model, sector, filling_fraction, level, ed_mode, ed_data
        )
        C_state = np.zeros((n_site, n_site), dtype=np.float64)
        density_a_state = np.zeros(n_site, dtype=np.float64)
        density_b_state = np.zeros(n_site, dtype=np.float64)
        _build_density_correlation_matrix(
            C_state,
            density_a_state,
            density_b_state,
            vector,
            basis,
            n_site,
            flavor_a,
            flavor_b,
            model.particle_statistics,
            subtract_disconnected=False,
        )
        C += weight * C_state
        density_a += weight * density_a_state
        density_b += weight * density_b_state

    # Connected correlator of rho = Σ_a w_a |ψ_a><ψ_a|.
    C -= np.outer(density_a, density_b)
    return C


def static_structure_factor_manifold_average(
    model: Real_Space_Second_Quantized_Model,
    manifold_states,
    *,
    weights=None,
    filling_fraction: Fraction,
    flavor_a: Callable[[int], bool] | None = None,
    flavor_b: Callable[[int], bool] | None = None,
    ed_mode: str = "matrix",
    ed_data: Symmetry_Resolved_ED_Data | None = None,
):
    """Connected ``S(q)`` of a normalized incoherent manifold projector.

    Julia counterpart: ``static_structure_factor_manifold_average``.

    With equal default weights this is basis independent inside a complete
    degenerate manifold and does not jump when its members merely exchange
    energy rank.

    Args:
        model: the second-quantized model.
        manifold_states: list of ``(sector, level)`` manifold states.
        weights: optional real weights (default: uniform).
        filling_fraction: particles per flattened vertex.
        flavor_a: boolean site filter for flavour a.
        flavor_b: boolean site filter for flavour b.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.

    Returns:
        tuple: ``(q_points, S_q)``.
    """
    C = _manifold_connected_density_correlation(
        model,
        manifold_states,
        weights=weights,
        filling_fraction=filling_fraction,
        flavor_a=flavor_a,
        flavor_b=flavor_b,
        ed_mode=ed_mode,
        ed_data=ed_data,
    )
    from tightbinding_py import initialize_uniform_grids_from_lattice

    kgrid = initialize_uniform_grids_from_lattice(model.lattice)
    phases = _precompute_phases(kgrid.site_cart_list, model.lattice.site_cart_list)
    S_q = np.zeros(kgrid.nsite, dtype=np.float64)
    _fourier_transform(S_q, phases, C, model.lattice.n_site)
    return list(kgrid.site_cart_list), S_q


def structure_factor_manifold_allowed_momenta(
    model: Real_Space_Second_Quantized_Model,
    manifold_states,
    *,
    weights=None,
    filling_fraction: Fraction,
    flavor_a: Callable[[int], bool] | None = None,
    flavor_b: Callable[[int], bool] | None = None,
    ed_mode: str = "matrix",
    ed_data: Symmetry_Resolved_ED_Data | None = None,
    fold_to_first_bz: bool = True,
):
    """Manifold-average ``S(q)`` restricted to the finite-torus allowed momenta.

    Julia counterpart: ``structure_factor_manifold_allowed_momenta``.

    Args:
        model: the second-quantized model.
        manifold_states: list of ``(sector, level)`` manifold states.
        weights: optional real weights (default: uniform).
        filling_fraction: particles per flattened vertex.
        flavor_a: boolean site filter for flavour a.
        flavor_b: boolean site filter for flavour b.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.
        fold_to_first_bz: fold momenta into the first BZ.

    Returns:
        tuple: ``(qx, qy, S_q)``.
    """
    q_points, S_q = static_structure_factor_manifold_average(
        model,
        manifold_states,
        weights=weights,
        filling_fraction=filling_fraction,
        flavor_a=flavor_a,
        flavor_b=flavor_b,
        ed_mode=ed_mode,
        ed_data=ed_data,
    )
    plot_points = (
        _fold_momenta_to_first_bz(q_points, model.lattice)
        if fold_to_first_bz
        else q_points
    )
    return [q[0] for q in plot_points], [q[1] for q in plot_points], S_q


def structure_factor_allowed_momenta(
    model: Real_Space_Second_Quantized_Model,
    sector_label,
    *,
    target_eigval_idx: int = 1,
    filling_fraction: Fraction,
    flavor_a: Callable[[int], bool] | None = None,
    flavor_b: Callable[[int], bool] | None = None,
    ed_mode: str = "matrix",
    ed_data: Symmetry_Resolved_ED_Data | None = None,
    fold_to_first_bz: bool = True,
):
    """Connected ``S(q)`` on the finite-torus allowed momenta, folded into the BZ.

    Julia counterpart: ``structure_factor_allowed_momenta``.

    Args:
        model: the second-quantized model.
        sector_label: ``"identity"`` or a momentum tuple.
        target_eigval_idx: **1-based** target eigenstate index.
        filling_fraction: particles per flattened vertex.
        flavor_a: boolean site filter for flavour a.
        flavor_b: boolean site filter for flavour b.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.
        fold_to_first_bz: fold momenta into the first BZ.

    Returns:
        tuple: ``(qx, qy, S_q)``.
    """
    q_points, S_q = static_structure_factor(
        model,
        sector_label,
        target_eigval_idx=target_eigval_idx,
        filling_fraction=filling_fraction,
        flavor_a=flavor_a,
        flavor_b=flavor_b,
        ed_mode=ed_mode,
        ed_data=ed_data,
    )
    plot_points = (
        _fold_momenta_to_first_bz(q_points, model.lattice)
        if fold_to_first_bz
        else q_points
    )
    qx = [q[0] for q in plot_points]
    qy = [q[1] for q in plot_points]
    return qx, qy, S_q


# ═══════════════════════════════════════════════════════════════════════════
# BZ heatmap: S(q) on a dense k-grid
# ═══════════════════════════════════════════════════════════════════════════


def compute_structure_factor_map(
    model: Real_Space_Second_Quantized_Model,
    sector_label,
    *,
    target_eigval_idx: int = 1,
    filling_fraction: Fraction,
    flavor_a: Callable[[int], bool] | None = None,
    flavor_b: Callable[[int], bool] | None = None,
    k_resolution: int = 61,
    ed_data: Symmetry_Resolved_ED_Data | None = None,
):
    """Dense connected ``S(q)`` map over ``[-1.5π, 1.5π]²``.

    Julia counterpart: ``compute_structure_factor_map``.

    Args:
        model: the second-quantized model.
        sector_label: ``"identity"`` or a momentum tuple.
        target_eigval_idx: **1-based** target eigenstate index.
        filling_fraction: particles per flattened vertex.
        flavor_a: boolean site filter for flavour a.
        flavor_b: boolean site filter for flavour b.
        k_resolution: number of grid points per direction.
        ed_data: optional precomputed ED data.

    Returns:
        tuple: ``(kx, ky, S_map)`` — the k-grid arrays and the
        ``(k_resolution, k_resolution)`` map with ``S_map[i, j] =
        S(kx[i], ky[j])`` (column-major, matching the Julia
        ``reshape``-of-``S_flat`` layout).
    """
    lattice = model.lattice
    n_site = lattice.n_site
    stats = model.particle_statistics

    if flavor_a is None:
        flavor_a = lambda i: True  # noqa: E731
    if flavor_b is None:
        flavor_b = lambda i: True  # noqa: E731

    ed_data, basis, c = _bootstrap_ed_sector(
        model, sector_label, filling_fraction, target_eigval_idx, "matrix", ed_data
    )

    C = np.zeros((n_site, n_site), dtype=np.float64)
    density_a = np.zeros(n_site, dtype=np.float64)
    density_b = np.zeros(n_site, dtype=np.float64)
    _build_density_correlation_matrix(
        C, density_a, density_b, c, basis, n_site, flavor_a, flavor_b, stats
    )

    span = 3 * np.pi
    kx = np.linspace(-span / 2, span / 2, num=k_resolution)
    ky = np.linspace(-span / 2, span / 2, num=k_resolution)
    q_points = [[x, y] for y in ky for x in kx]

    phases = _precompute_phases(q_points, lattice.site_cart_list)
    S_flat = np.zeros(len(q_points), dtype=np.float64)
    _fourier_transform(S_flat, phases, C, n_site)

    S_map = S_flat.reshape(k_resolution, k_resolution, order="F")
    return kx, ky, S_map


def compute_structure_factor_manifold_average_map(
    model: Real_Space_Second_Quantized_Model,
    manifold_states,
    *,
    weights=None,
    filling_fraction: Fraction,
    flavor_a: Callable[[int], bool] | None = None,
    flavor_b: Callable[[int], bool] | None = None,
    k_resolution: int = 61,
    ed_mode: str = "matrix",
    ed_data: Symmetry_Resolved_ED_Data | None = None,
):
    """Dense manifold-average connected ``S(q)`` map over ``[-1.5π, 1.5π]²``.

    Julia counterpart: ``compute_structure_factor_manifold_average_map``.

    Args:
        model: the second-quantized model.
        manifold_states: list of ``(sector, level)`` manifold states.
        weights: optional real weights (default: uniform).
        filling_fraction: particles per flattened vertex.
        flavor_a: boolean site filter for flavour a.
        flavor_b: boolean site filter for flavour b.
        k_resolution: number of grid points per direction.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.

    Returns:
        tuple: ``(kx, ky, S_map)``.
    """
    C = _manifold_connected_density_correlation(
        model,
        manifold_states,
        weights=weights,
        filling_fraction=filling_fraction,
        flavor_a=flavor_a,
        flavor_b=flavor_b,
        ed_mode=ed_mode,
        ed_data=ed_data,
    )
    span = 3 * np.pi
    kx = np.linspace(-span / 2, span / 2, num=k_resolution)
    ky = np.linspace(-span / 2, span / 2, num=k_resolution)
    q_points = [[x, y] for y in ky for x in kx]
    phases = _precompute_phases(q_points, model.lattice.site_cart_list)
    S_flat = np.zeros(len(q_points), dtype=np.float64)
    _fourier_transform(S_flat, phases, C, model.lattice.n_site)
    return kx, ky, S_flat.reshape(k_resolution, k_resolution, order="F")


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


def plot_structure_factor_map(
    model: Real_Space_Second_Quantized_Model,
    sector_label,
    *,
    target_eigval_idx: int = 1,
    filling_fraction: Fraction,
    flavor_a: Callable[[int], bool] | None = None,
    flavor_b: Callable[[int], bool] | None = None,
    k_resolution: int = 61,
    ed_data: Symmetry_Resolved_ED_Data | None = None,
    fig_path: str | None = None,
    title: str = "Connected static structure factor S(q)",
):
    """Compute and plot the dense connected ``S(q)`` map with the BZ boundary.

    Julia counterpart: ``plot_structure_factor_map`` (CairoMakie → matplotlib).

    Args:
        model: the second-quantized model.
        sector_label: ``"identity"`` or a momentum tuple.
        target_eigval_idx: **1-based** target eigenstate index.
        filling_fraction: particles per flattened vertex.
        flavor_a: boolean site filter for flavour a.
        flavor_b: boolean site filter for flavour b.
        k_resolution: number of grid points per direction.
        ed_data: optional precomputed ED data.
        fig_path: optional path to save the figure.
        title: figure title.

    Returns:
        tuple: ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    kx, ky, S_map = compute_structure_factor_map(
        model,
        sector_label,
        target_eigval_idx=target_eigval_idx,
        filling_fraction=filling_fraction,
        flavor_a=flavor_a,
        flavor_b=flavor_b,
        k_resolution=k_resolution,
        ed_data=ed_data,
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax.set_xlabel("k_x")
    ax.set_ylabel("k_y")
    ax.set_title(title)
    ax.set_aspect("equal")
    hm = ax.pcolormesh(kx, ky, S_map.T, shading="auto", cmap="viridis")
    _draw_bz_boundary(ax, model.lattice)
    fig.colorbar(hm, ax=ax, label="S(q)")

    if fig_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
        fig.savefig(fig_path)
    return fig, ax


def plot_structure_factor_allowed_momenta(
    model: Real_Space_Second_Quantized_Model,
    sector_label,
    *,
    target_eigval_idx: int = 1,
    filling_fraction: Fraction,
    flavor_a: Callable[[int], bool] | None = None,
    flavor_b: Callable[[int], bool] | None = None,
    ed_mode: str = "matrix",
    ed_data: Symmetry_Resolved_ED_Data | None = None,
    fold_to_first_bz: bool = True,
    fig_path: str | None = None,
    title: str = "Allowed-momentum S(q)",
    markersize: float = 24,
    colorrange=None,
):
    """Plot ``S(q)`` at the finite-torus allowed momenta over the BZ boundary.

    Julia counterpart: ``plot_structure_factor_allowed_momenta``.

    Args:
        model: the second-quantized model.
        sector_label: ``"identity"`` or a momentum tuple.
        target_eigval_idx: **1-based** target eigenstate index.
        filling_fraction: particles per flattened vertex.
        flavor_a: boolean site filter for flavour a.
        flavor_b: boolean site filter for flavour b.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.
        fold_to_first_bz: fold momenta into the first BZ.
        fig_path: optional path to save the figure.
        title: figure title.
        markersize: marker size.
        colorrange: optional ``(vmin, vmax)`` color range.

    Returns:
        tuple: ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    qx, qy, S_q = structure_factor_allowed_momenta(
        model,
        sector_label,
        target_eigval_idx=target_eigval_idx,
        filling_fraction=filling_fraction,
        flavor_a=flavor_a,
        flavor_b=flavor_b,
        ed_mode=ed_mode,
        ed_data=ed_data,
        fold_to_first_bz=fold_to_first_bz,
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax.set_xlabel("q_x")
    ax.set_ylabel("q_y")
    ax.set_title(title)
    ax.set_aspect("equal")
    kwargs = {"s": markersize, "c": S_q, "cmap": "viridis"}
    if colorrange is not None:
        kwargs["vmin"] = colorrange[0]
        kwargs["vmax"] = colorrange[1]
    sc = ax.scatter(qx, qy, **kwargs)
    _draw_bz_boundary(ax, model.lattice)
    fig.colorbar(sc, ax=ax, label="S(q)")

    if fig_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
        fig.savefig(fig_path)
    return fig, ax


def plot_structure_factor_map_panels(
    maps,
    *,
    fig_path: str | None = None,
    title: str = "Connected static structure factor S(q)",
):
    """Plot several precomputed ``S(q)`` maps in one row with a unified scale.

    Julia counterpart: ``plot_structure_factor_map_panels``.

    Each entry of ``maps`` exposes fields ``kx``, ``ky``, ``values``,
    ``lattice`` and ``title`` (e.g. a :class:`types.SimpleNamespace`).

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
    fig.colorbar(hm_ref, ax=axes, label="S(q)")

    if fig_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
        fig.savefig(fig_path)
    return fig, axes


def plot_structure_factor_allowed_momenta_panels(
    maps,
    *,
    fig_path: str | None = None,
    title: str = "Allowed-momentum S(q)",
    markersize: float = 24,
):
    """Plot several allowed-momentum ``S(q)`` datasets in one row.

    Julia counterpart: ``plot_structure_factor_allowed_momenta_panels``.

    Each entry of ``maps`` exposes fields ``qx``, ``qy``, ``values``,
    ``lattice`` and ``title``.

    Args:
        maps: list of dataset entries.
        fig_path: optional path to save the figure.
        title: overall figure title.
        markersize: marker size.

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
    sc_ref = None
    for idx, m in enumerate(maps):
        ax = axes[idx]
        ax.set_xlabel("q_x")
        ax.set_ylabel("q_y")
        ax.set_title(m.title)
        ax.set_aspect("equal")
        sc = ax.scatter(
            m.qx, m.qy, s=markersize, c=m.values, cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        _draw_bz_boundary(ax, m.lattice)
        if sc_ref is None:
            sc_ref = sc
    fig.suptitle(title, fontsize=18)
    fig.colorbar(sc_ref, ax=axes, label="S(q)")

    if fig_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
        fig.savefig(fig_path)
    return fig, axes


__all__ = [
    "static_structure_factor",
    "structure_factor_allowed_momenta",
    "static_structure_factor_manifold_average",
    "structure_factor_manifold_allowed_momenta",
    "compute_structure_factor_map",
    "compute_structure_factor_manifold_average_map",
    "plot_structure_factor_map",
    "plot_structure_factor_map_panels",
    "plot_structure_factor_allowed_momenta",
    "plot_structure_factor_allowed_momenta_panels",
]
