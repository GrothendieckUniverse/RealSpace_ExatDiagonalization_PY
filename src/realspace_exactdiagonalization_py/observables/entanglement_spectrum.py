"""Spatial-orbital and momentum-resolved particle entanglement spectra.

Faithful port of ``observables/entanglement_spectrum.jl`` from the Julia
package ``RealSpace_ExactDiagonalization.jl``.

A symmetry-sector eigenvector is first expanded into Fock amplitudes, then
Schmidt blocks at fixed particle number in region A are formed.  The spatial
entanglement spectrum returns both Schmidt probabilities and entanglement
energies :math:`\\xi = -\\log(\\lambda)`; the momentum-resolved particle
entanglement spectrum (PES) traces out :math:`N_B = N - N_A` particles and
block-diagonalizes the reduced density matrix in the many-body momentum
sectors of subsystem A.
"""

from __future__ import annotations

import math
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from ..utils.bitwise_operations import COMPLEX_ONE, n_occupied_for_mask
from ..second_quantized_model import (
    Particle_Statistics,
    Real_Space_Second_Quantized_Model,
)
from ..symmetry_resolved_ed import (
    Symmetry_Resolved_ED_Data,
    Symmetry_Sector_Basis,
    _find_irrep_idx,
    _first_combination_mask,
    _gosper_next,
    apply_operation_to_mask,
    build_symmetry_orbit_catalog,
    build_symmetry_sector_basis,
    build_translation_group,
    build_translation_irrep_list,
    group_order,
)
from ..observables.charge_pump import _normalize_manifold_state_specs
from .static_structure_factor import _bootstrap_ed_sector


# ═══════════════════════════════════════════════════════════════════════════
# Sector eigenvector resolution & Fock-amplitude expansion
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_sector_eigenvector(
    model: Real_Space_Second_Quantized_Model,
    sector_label,
    *,
    target_eigval_idx: int,
    filling_fraction: Fraction,
    ed_mode: str,
    ed_data: Symmetry_Resolved_ED_Data | None,
):
    """Resolve a symmetry-sector eigenvector to ``(ed_data, basis, c)``.

    Julia counterpart: ``_resolve_sector_eigenvector``.  The sector label is
    normalized (``"identity"`` stays ``"identity"``, anything else becomes a
    tuple of ints) and the shared :func:`_bootstrap_ed_sector` is invoked.

    Args:
        model: the second-quantized model.
        sector_label: ``"identity"`` or a momentum tuple.
        target_eigval_idx: **1-based** target eigenstate index.
        filling_fraction: particles per flattened vertex.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.

    Returns:
        tuple: ``(ed_data, basis, c)``.
    """
    label = "identity" if sector_label == "identity" else tuple(int(x) for x in sector_label)
    return _bootstrap_ed_sector(
        model, label, filling_fraction, target_eigval_idx, ed_mode, ed_data
    )


def _expand_sector_state_to_fock_amplitudes(
    c: np.ndarray,
    basis: Symmetry_Sector_Basis,
    particle_statistics: Particle_Statistics,
    *,
    atol: float = 1e-12,
) -> dict[int, complex]:
    """Expand a symmetry-sector eigenvector into (normalized) Fock amplitudes.

    Julia counterpart: ``_expand_sector_state_to_fock_amplitudes``.

    The normalized projected basis is
    :math:`|[s];\\chi\\rangle = |G\\,|\\mathrm{Stab}(s)|^{-1/2}
    \\sum_g \\chi(g)^* U_g |s\\rangle`, so each representative ``s`` contributes

    .. math::
        \\psi(m) = \\sum_{col} \\frac{c_{col}}{\\sqrt{|G|\\,|\\mathrm{Stab}(col)|}}
        \\sum_g \\chi(g)^* \\, \\alpha_g(s_{col})\\, U_g |s_{col}\\rangle.

    Amplitudes below ``atol`` are pruned and the result renormalized to unit norm.

    Args:
        c: sector eigenvector amplitudes.
        basis: the symmetry-sector basis.
        particle_statistics: particle statistics of the model.
        atol: amplitude prune threshold.

    Returns:
        dict[int, complex]: ``mask → amplitude`` map.
    """
    G = basis.symmetry_group
    nG = group_order(G)
    amplitudes: dict[int, complex] = {}

    for col, repr_mask in enumerate(basis.representative_mask_list):
        coeff = c[col]
        if abs(coeff) <= atol:
            continue
        prefactor = coeff / np.sqrt(nG * basis.stabilizer_order_list[col])

        for gidx, op in enumerate(G.operations):
            shifted, phase = apply_operation_to_mask(
                repr_mask, op, particle_statistics
            )
            amp = prefactor * np.conj(basis.irrep.values[gidx]) * phase
            amplitudes[shifted] = amplitudes.get(shifted, 0.0 + 0.0j) + amp

    for mask in list(amplitudes.keys()):
        if abs(amplitudes[mask]) <= atol:
            del amplitudes[mask]

    norm2 = sum(abs(a) ** 2 for a in amplitudes.values())
    if norm2 <= atol:
        raise ValueError("Expanded state has near-zero norm.")
    invnorm = 1.0 / np.sqrt(norm2)
    for mask in amplitudes:
        amplitudes[mask] *= invnorm
    return amplitudes


# ═══════════════════════════════════════════════════════════════════════════
# Spatial-orbital bipartition
# ═══════════════════════════════════════════════════════════════════════════


def _subsystem_maps(partition_a, n_site: int):
    """Build the two subsystem position maps from a bipartition.

    Julia counterpart: ``_subsystem_maps``.

    Args:
        partition_a: 1-based flattened site indices of subsystem A.
        n_site: total number of sites.

    Returns:
        tuple: ``(partition_a, partition_b, a_pos, b_pos)`` where ``a_pos`` /
        ``b_pos`` are length-``n_site`` arrays whose ``site-1`` entry is the
        **1-based** position of ``site`` in its subsystem (0 if absent).
    """
    partition_a = [int(site) for site in partition_a]
    seen = [False] * n_site
    for site in partition_a:
        if not (1 <= site <= n_site):
            raise ValueError(f"partition_a contains invalid site index {site}.")
        if seen[site - 1]:
            raise ValueError(f"partition_a contains duplicate site index {site}.")
        seen[site - 1] = True

    partition_b = [site for site in range(1, n_site + 1) if not seen[site - 1]]
    a_pos = [0] * n_site
    b_pos = [0] * n_site
    for idx, site in enumerate(partition_a):
        a_pos[site - 1] = idx + 1
    for idx, site in enumerate(partition_b):
        b_pos[site - 1] = idx + 1
    return partition_a, partition_b, a_pos, b_pos


def _split_mask_for_partition(mask: int, a_pos, b_pos):
    """Split a full mask into subsystem masks plus the A particle number.

    Julia counterpart: ``_split_mask_for_partition``.

    Args:
        mask: the full occupation mask.
        a_pos: subsystem-A position map from :func:`_subsystem_maps`.
        b_pos: subsystem-B position map from :func:`_subsystem_maps`.

    Returns:
        tuple: ``(mask_a, mask_b, n_a)``.
    """
    mask_a = 0
    mask_b = 0
    n_a = 0

    tmp = mask
    while tmp != 0:
        lsb = tmp & -tmp
        site = (lsb.bit_length() - 1) + 1
        if a_pos[site - 1] != 0:
            mask_a |= 1 << (a_pos[site - 1] - 1)
            n_a += 1
        else:
            mask_b |= 1 << (b_pos[site - 1] - 1)
        tmp ^= lsb

    return mask_a, mask_b, n_a


def _partition_sign(
    particle_statistics: Particle_Statistics, mask: int, is_a_site
) -> complex:
    """Fermionic parity needed to order all A creation operators before all B's.

    Julia counterpart: ``_partition_sign`` (bosonic ⇒ ``+1``).

    Args:
        particle_statistics: particle statistics of the model.
        mask: the full occupation mask.
        is_a_site: boolean array with ``is_a_site[site-1] == True`` iff
            ``site`` belongs to subsystem A.

    Returns:
        complex: ``+1`` or ``-1``.
    """
    if particle_statistics is not Particle_Statistics.FERMIONIC:
        return COMPLEX_ONE

    b_before = 0
    swaps = 0
    tmp = mask
    while tmp != 0:
        lsb = tmp & -tmp
        site = (lsb.bit_length() - 1) + 1
        if is_a_site[site - 1]:
            swaps += b_before
        else:
            b_before += 1
        tmp ^= lsb
    return -COMPLEX_ONE if swaps % 2 == 1 else COMPLEX_ONE


def _schmidt_blocks_from_amplitudes(
    amplitudes: dict[int, complex],
    partition_a,
    n_site: int,
    particle_statistics: Particle_Statistics,
):
    """Group Fock amplitudes into Schmidt blocks by particle number in A.

    Julia counterpart: ``_schmidt_blocks_from_amplitudes``.

    Args:
        amplitudes: ``mask → amplitude`` map.
        partition_a: subsystem A site indices.
        n_site: total number of sites.
        particle_statistics: particle statistics of the model.

    Returns:
        dict[int, list]: ``n_a → [(mask_a, mask_b, signed_amp), ...]``.
    """
    part_a, _, a_pos, b_pos = _subsystem_maps(partition_a, n_site)
    is_a_site = [False] * n_site
    for site in part_a:
        is_a_site[site - 1] = True

    block_entries: dict[int, list] = {}
    for mask, amp in amplitudes.items():
        mask_a, mask_b, n_a = _split_mask_for_partition(mask, a_pos, b_pos)
        signed_amp = _partition_sign(particle_statistics, mask, is_a_site) * amp
        block_entries.setdefault(n_a, []).append((mask_a, mask_b, signed_amp))
    return block_entries


def _entanglement_probabilities_by_block(block_entries):
    """Schmidt probabilities and entanglement energies from Schmidt blocks.

    Julia counterpart: ``_entanglement_probabilities_by_block``.

    Args:
        block_entries: ``n_a → [(mask_a, mask_b, amp), ...]`` map.

    Returns:
        tuple: ``(levels, probabilities)`` — ``levels`` is a list of
        :class:`types.SimpleNamespace` rows sorted by entanglement energy, and
        ``probabilities`` is the globally descending list of Schmidt values.
    """
    rows: list[SimpleNamespace] = []
    probabilities: list[float] = []

    for n_a in sorted(block_entries.keys()):
        entries = block_entries[n_a]
        a_masks = sorted({e[0] for e in entries})
        b_masks = sorted({e[1] for e in entries})
        a_index = {m: idx for idx, m in enumerate(a_masks)}
        b_index = {m: idx for idx, m in enumerate(b_masks)}
        M = np.zeros((len(a_masks), len(b_masks)), dtype=np.complex128)

        for (mask_a, mask_b, amp) in entries:
            M[a_index[mask_a], b_index[mask_b]] += amp

        s = np.linalg.svd(M, compute_uv=False)
        lam = (np.real(s) ** 2).tolist()
        lam = [x for x in lam if x > 1e-14]
        lam = sorted(lam, reverse=True)
        probabilities.extend(lam)
        for level, p in enumerate(lam, start=1):
            rows.append(
                SimpleNamespace(
                    n_a=n_a,
                    level=level,
                    probability=p,
                    entanglement_energy=-math.log(p),
                    dim_a=len(a_masks),
                    dim_b=len(b_masks),
                )
            )

    rows.sort(key=lambda row: row.entanglement_energy)
    return rows, sorted(probabilities, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# Public API — spatial orbital entanglement spectrum
# ═══════════════════════════════════════════════════════════════════════════


def entanglement_spectrum(
    model: Real_Space_Second_Quantized_Model,
    sector_label,
    *,
    partition_a,
    target_eigval_idx: int = 1,
    filling_fraction: Fraction,
    ed_mode: str = "matrix",
    ed_data: Symmetry_Resolved_ED_Data | None = None,
) -> SimpleNamespace:
    """Spatial orbital entanglement spectrum of one many-body eigenstate.

    Julia counterpart: ``entanglement_spectrum``.

    Args:
        model: the second-quantized model.
        sector_label: ``"identity"`` or a momentum tuple.
        partition_a: 1-based flattened site indices assigned to subsystem A;
            subsystem B is its complement.
        target_eigval_idx: **1-based** target eigenstate index.
        filling_fraction: particles per flattened vertex.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.

    Returns:
        SimpleNamespace: ``levels``, ``probabilities``,
        ``entanglement_energies``, ``partition_a``, ``partition_b``,
        ``sector_label``, ``target_eigval_idx``, ``filling_fraction``,
        ``norm_probability``, ``ed_data``.
    """
    ed_data, basis, c = _resolve_sector_eigenvector(
        model,
        sector_label,
        target_eigval_idx=target_eigval_idx,
        filling_fraction=filling_fraction,
        ed_mode=ed_mode,
        ed_data=ed_data,
    )

    amplitudes = _expand_sector_state_to_fock_amplitudes(
        c, basis, model.particle_statistics
    )
    blocks = _schmidt_blocks_from_amplitudes(
        amplitudes, partition_a, model.lattice.n_site, model.particle_statistics
    )
    levels, probabilities = _entanglement_probabilities_by_block(blocks)

    return SimpleNamespace(
        levels=levels,
        probabilities=probabilities,
        entanglement_energies=[row.entanglement_energy for row in levels],
        partition_a=[int(x) for x in partition_a],
        partition_b=[
            site
            for site in range(1, model.lattice.n_site + 1)
            if site not in partition_a
        ],
        sector_label=(
            "identity" if sector_label == "identity" else tuple(int(x) for x in sector_label)
        ),
        target_eigval_idx=target_eigval_idx,
        filling_fraction=filling_fraction,
        norm_probability=sum(probabilities),
        ed_data=ed_data,
    )


def plot_entanglement_spectrum(
    result,
    *,
    fig_path: str | None = None,
    title: str = "Entanglement spectrum",
):
    """Scatter plot of :math:`\\xi=-\\log(\\lambda)` grouped by :math:`N_A`.

    Julia counterpart: ``plot_entanglement_spectrum`` (CairoMakie → matplotlib).

    Args:
        result: the result of :func:`entanglement_spectrum`.
        fig_path: optional path to save the figure.
        title: figure title.

    Returns:
        tuple: ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlabel("N_A")
    ax.set_ylabel("ξ = -log(λ)")
    ax.set_title(title)
    xs = [row.n_a for row in result.levels]
    ys = [row.entanglement_energy for row in result.levels]
    ax.scatter(xs, ys, s=8, c=ys, cmap="viridis")

    if fig_path is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
        fig.savefig(fig_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# Momentum-resolved particle entanglement spectrum
# ═══════════════════════════════════════════════════════════════════════════


def _fixed_particle_masks(n_site: int, n_filled: int) -> list[int]:
    """All masks with exactly ``n_filled`` set bits among ``n_site`` bits.

    Julia counterpart: ``_fixed_particle_masks`` (Gosper's hack enumeration).

    Args:
        n_site: total number of sites.
        n_filled: number of set bits.

    Returns:
        list[int]: the masks in lexicographic order.
    """
    masks: list[int] = []
    if n_filled == 0:
        masks.append(0)
        return masks
    x = _first_combination_mask(n_filled)
    upper = 1 << n_site
    while x < upper:
        masks.append(x)
        x = _gosper_next(x)
    return masks


def _occupied_sites(mask: int) -> list[int]:
    """1-based occupied site indices of a mask (ascending order).

    Julia counterpart: ``_occupied_sites``.

    Args:
        mask: the occupation mask.

    Returns:
        list[int]: occupied site indices.
    """
    sites: list[int] = []
    tmp = mask
    while tmp != 0:
        lsb = tmp & -tmp
        sites.append((lsb.bit_length() - 1) + 1)
        tmp ^= lsb
    return sites


def _particle_submasks(mask: int, n_take: int) -> list[int]:
    """All submasks of ``mask`` with exactly ``n_take`` of its occupied bits.

    Julia counterpart: ``_particle_submasks`` (recursive combination enumeration).

    Args:
        mask: the full mask.
        n_take: number of occupied bits to keep.

    Returns:
        list[int]: the submasks.
    """
    sites = _occupied_sites(mask)
    out: list[int] = []

    def rec(start: int, left: int, acc: int) -> None:
        if left == 0:
            out.append(acc)
            return
        max_start = len(sites) - left + 1  # 1-based inclusive bound
        for idx in range(start, max_start + 1):
            rec(idx + 1, left - 1, acc | (1 << (sites[idx - 1] - 1)))

    rec(1, n_take, 0)
    return out


def _particle_partition_sign(
    particle_statistics: Particle_Statistics, full_mask: int, mask_a: int
) -> complex:
    """Fermionic parity to separate subsystem-A and subsystem-B particles.

    Julia counterpart: ``_particle_partition_sign`` (bosonic ⇒ ``+1``).

    Args:
        particle_statistics: particle statistics of the model.
        full_mask: the full occupation mask.
        mask_a: the subsystem-A submask.

    Returns:
        complex: ``+1`` or ``-1``.
    """
    if particle_statistics is not Particle_Statistics.FERMIONIC:
        return COMPLEX_ONE

    b_before = 0
    swaps = 0
    tmp = full_mask
    while tmp != 0:
        lsb = tmp & -tmp
        if (mask_a & lsb) != 0:
            swaps += b_before
        else:
            b_before += 1
        tmp ^= lsb
    return -COMPLEX_ONE if swaps % 2 == 1 else COMPLEX_ONE


def _particle_partition_matrix(
    amplitudes: dict[int, complex],
    n_site: int,
    n_particles_a: int,
    n_particles_b: int,
    particle_statistics: Particle_Statistics,
    basis_a_index: dict[int, int],
    basis_b_index: dict[int, int],
) -> np.ndarray:
    """Particle-bipartition coefficient matrix of the Fock amplitudes.

    Julia counterpart: ``_particle_partition_matrix``.

    Args:
        amplitudes: ``mask → amplitude`` map.
        n_site: total number of sites.
        n_particles_a: particle number kept in subsystem A.
        n_particles_b: particle number traced out (subsystem B).
        particle_statistics: particle statistics of the model.
        basis_a_index: ``mask_a → index`` map (0-based).
        basis_b_index: ``mask_b → index`` map (0-based).

    Returns:
        np.ndarray: complex ``(dim_a, dim_b)`` matrix.
    """
    M = np.zeros((len(basis_a_index), len(basis_b_index)), dtype=np.complex128)
    for full_mask, amp in amplitudes.items():
        if n_occupied_for_mask(full_mask) != n_particles_a + n_particles_b:
            continue
        for mask_a in _particle_submasks(full_mask, n_particles_a):
            mask_b = full_mask ^ mask_a
            ia = basis_a_index[mask_a]
            ib = basis_b_index[mask_b]
            M[ia, ib] += _particle_partition_sign(
                particle_statistics, full_mask, mask_a
            ) * amp
    return M


def _resolve_many_body_manifold_states(
    model: Real_Space_Second_Quantized_Model,
    sector_labels,
    *,
    filling_fraction: Fraction,
    target_eigval_idx: int,
    manifold_states,
    ed_mode: str,
    ed_data: Symmetry_Resolved_ED_Data | None,
):
    """Expand every state of a many-body manifold into Fock amplitudes.

    Julia counterpart: ``_resolve_many_body_manifold_states``.

    Args:
        model: the second-quantized model.
        sector_labels: ``"identity"`` or a list of momentum tuples.
        filling_fraction: particles per flattened vertex.
        target_eigval_idx: **1-based** default level.
        manifold_states: optional explicit ``(sector, level)`` states.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.

    Returns:
        tuple: ``(labels, state_specs, states, energies, ed_data)``.
    """
    if manifold_states is None:
        explicit_states = [
            (tuple(int(x) for x in label), target_eigval_idx)
            for label in sector_labels
        ]
    else:
        explicit_states = manifold_states
    labels, state_specs, _, _ = _normalize_manifold_state_specs(
        sector_labels, 1, explicit_states
    )

    states: list[dict[int, complex]] = []
    energies: list[float] = []
    for state in state_specs:
        label, level = state.sector, state.level
        ed_data, basis, c = _resolve_sector_eigenvector(
            model,
            label,
            target_eigval_idx=level,
            filling_fraction=filling_fraction,
            ed_mode=ed_mode,
            ed_data=ed_data,
        )
        states.append(
            _expand_sector_state_to_fock_amplitudes(c, basis, model.particle_statistics)
        )
        irrep_idx = _find_irrep_idx(ed_data, label)
        vals = ed_data.ed_scan_res[irrep_idx][0]
        energies.append(float(vals[level - 1]))
    return labels, state_specs, states, energies, ed_data


def _sector_embedding_matrix(
    full_basis_index: dict[int, int],
    sector_basis: Symmetry_Sector_Basis,
    particle_statistics: Particle_Statistics,
) -> np.ndarray:
    """Embedding of a symmetry sector into the full Fock basis.

    Julia counterpart: ``_sector_embedding_matrix``.

    Args:
        full_basis_index: ``mask → index`` map of the full basis.
        sector_basis: the symmetry-sector basis.
        particle_statistics: particle statistics of the model.

    Returns:
        np.ndarray: complex ``(dim_full, dim_sector)`` matrix ``U``.
    """
    dim_full = len(full_basis_index)
    dim_sector = len(sector_basis.representative_mask_list)
    U = np.zeros((dim_full, dim_sector), dtype=np.complex128)
    for col in range(dim_sector):
        c = np.zeros(dim_sector, dtype=np.complex128)
        c[col] = 1.0
        amps = _expand_sector_state_to_fock_amplitudes(
            c, sector_basis, particle_statistics
        )
        for mask, amp in amps.items():
            U[full_basis_index[mask], col] = amp
    return U


def particle_entanglement_spectrum(
    model: Real_Space_Second_Quantized_Model,
    sector_labels,
    *,
    n_particles_a: int,
    filling_fraction: Fraction,
    target_eigval_idx: int = 1,
    manifold_states=None,
    ed_mode: str = "matrix",
    ed_data: Symmetry_Resolved_ED_Data | None = None,
    probability_cutoff: float = 1e-14,
) -> SimpleNamespace:
    """Translation-symmetry-resolved particle entanglement spectrum (PES).

    Julia counterpart: ``particle_entanglement_spectrum``.  The reduced density
    matrix is built by tracing out :math:`N_B = N - N_A` particles and is then
    block-diagonalized in the many-body momentum sectors of subsystem A.

    Args:
        model: the second-quantized model.
        sector_labels: ``"identity"`` or a list of momentum tuples.
        n_particles_a: particle number kept in subsystem A.
        filling_fraction: particles per flattened vertex.
        target_eigval_idx: **1-based** default level for each sector.
        manifold_states: optional explicit ``(sector, level)`` states.
        ed_mode: ED mode.
        ed_data: optional precomputed ED data.
        probability_cutoff: threshold below which eigenvalues are dropped.

    Returns:
        SimpleNamespace: ``levels``, ``sector_summaries``, ``n_particles_a``,
        ``n_particles_b``, ``n_particles_total``, ``sector_labels``,
        ``manifold_states``, ``manifold_energies``, ``norm_probability``,
        ``filling_fraction``, ``ed_data``.
    """
    n_site = model.lattice.n_site
    filling_fraction = Fraction(filling_fraction)
    n_particles_total = int(filling_fraction * n_site)
    if not (0 <= n_particles_a <= n_particles_total):
        raise ValueError(
            f"n_particles_a must be between 0 and total particle number "
            f"{n_particles_total}."
        )
    n_particles_b = n_particles_total - n_particles_a

    labels, state_specs, states, energies, ed_data = _resolve_many_body_manifold_states(
        model,
        sector_labels,
        filling_fraction=filling_fraction,
        target_eigval_idx=target_eigval_idx,
        manifold_states=manifold_states,
        ed_mode=ed_mode,
        ed_data=ed_data,
    )

    basis_a = _fixed_particle_masks(n_site, n_particles_a)
    basis_b = _fixed_particle_masks(n_site, n_particles_b)
    index_a = {mask: idx for idx, mask in enumerate(basis_a)}
    index_b = {mask: idx for idx, mask in enumerate(basis_b)}

    rho = np.zeros((len(basis_a), len(basis_a)), dtype=np.complex128)
    for amps in states:
        M = _particle_partition_matrix(
            amps,
            n_site,
            n_particles_a,
            n_particles_b,
            model.particle_statistics,
            index_a,
            index_b,
        )
        rho += (M @ M.conj().T) / len(states)
    tr_rho = float(np.real(np.trace(rho)))
    if tr_rho <= probability_cutoff:
        raise ValueError("Particle reduced density matrix has near-zero trace.")
    rho /= tr_rho

    G = build_translation_group(model.lattice)
    catalog_a = build_symmetry_orbit_catalog(
        second_quantized_model=model,
        n_filled=n_particles_a,
        symmetry_group=G,
        particle_statistics=model.particle_statistics,
    )
    irrep_list = build_translation_irrep_list(G, model.lattice)

    levels: list[SimpleNamespace] = []
    sector_summaries: list[SimpleNamespace] = []
    for irrep in irrep_list:
        sector_basis = build_symmetry_sector_basis(catalog_a, irrep)
        dim_sector = len(sector_basis.representative_mask_list)
        if dim_sector == 0:
            continue
        U = _sector_embedding_matrix(index_a, sector_basis, model.particle_statistics)
        Uk = U.conj().T @ rho @ U
        Uk = (Uk + Uk.conj().T) / 2
        vals = np.linalg.eigvalsh(Uk)
        vals = vals[::-1]  # descending
        vals = vals[vals > probability_cutoff]
        for level, p in enumerate(vals, start=1):
            levels.append(
                SimpleNamespace(
                    momentum=tuple(int(x) for x in irrep.label),
                    level=level,
                    probability=float(p),
                    entanglement_energy=-math.log(float(p)),
                    sector_dim=dim_sector,
                )
            )
        sector_summaries.append(
            SimpleNamespace(
                momentum=tuple(int(x) for x in irrep.label),
                sector_dim=dim_sector,
                kept_levels=len(vals),
                weight=float(sum(vals)),
            )
        )
    levels.sort(key=lambda row: row.entanglement_energy)

    return SimpleNamespace(
        levels=levels,
        sector_summaries=sector_summaries,
        n_particles_a=n_particles_a,
        n_particles_b=n_particles_b,
        n_particles_total=n_particles_total,
        sector_labels=labels,
        manifold_states=state_specs,
        manifold_energies=energies,
        norm_probability=sum(row.probability for row in levels),
        filling_fraction=filling_fraction,
        ed_data=ed_data,
    )


def plot_particle_entanglement_spectrum(
    result,
    *,
    fig_path: str | None = None,
    title: str = "Momentum-resolved particle entanglement spectrum",
):
    """Scatter plot of the momentum-resolved PES.

    Julia counterpart: ``plot_particle_entanglement_spectrum``.

    Args:
        result: the result of :func:`particle_entanglement_spectrum`.
        fig_path: optional path to save the figure.
        title: figure title.

    Returns:
        tuple: ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    momenta = sorted(set(row.momentum for row in result.levels))
    momentum_index = {k: idx for idx, k in enumerate(momenta)}
    xs = [momentum_index[row.momentum] for row in result.levels]
    ys = [row.entanglement_energy for row in result.levels]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.set_xlabel("subsystem momentum sector")
    ax.set_ylabel("ξ = -log(λ)")
    ax.set_title(title)
    ax.set_xticks(range(len(momenta)))
    ax.set_xticklabels([repr(k) for k in momenta], rotation=60)
    ax.scatter(xs, ys, s=7, c=ys, cmap="viridis")

    if fig_path is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
        fig.savefig(fig_path)
    return fig, ax


__all__ = [
    "entanglement_spectrum",
    "plot_entanglement_spectrum",
    "particle_entanglement_spectrum",
    "plot_particle_entanglement_spectrum",
]
