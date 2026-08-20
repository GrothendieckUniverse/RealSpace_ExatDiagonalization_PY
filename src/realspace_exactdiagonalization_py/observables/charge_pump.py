"""One-dimensional flux insertion and many-body polarization.

Faithful port of ``observables/charge_pump.jl`` from the Julia package
``RealSpace_ExactDiagonalization.jl`` — the finite-size analogue of the
Laughlin–Thouless charge pump: thread a flux :math:`\\theta` through one
periodic direction of a torus and track the winding of the many-body
polarization (Resta's periodic position operator) in the transverse
direction.
"""

from __future__ import annotations

import cmath
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy.sparse as sp

from ..utils.bitwise_operations import COMPLEX_ONE, Mask
from ..second_quantized_model import (
    Particle_Statistics,
    Real_Space_Second_Quantized_Model,
)
from ..symmetry_resolved_ed import (
    CanonicalMap,
    Symmetry_Sector_Basis,
    apply_operation_to_mask,
    build_ed_data,
    build_symmetry_sector_basis,
    build_translation_group,
    ed_scan,
    load_checkpoint,
    group_order,
)


def many_body_position_phases(
    lattice, direction: int, *, include_sublattice: bool = True
) -> np.ndarray:
    """Single-site phase factors of Resta's periodic position operator.

    Julia counterpart: ``many_body_position_phases``.  The operator in
    direction α is

    .. math::
        U_\\alpha = \\exp\\Big(\\frac{2\\pi i}{L_\\alpha}
        \\sum_j x_{j,\\alpha}\\,\\hat n_j\\Big),

    where :math:`x_{j,\\alpha}` is the α-component of the *crystal*
    coordinate of site :math:`j`, i.e. ``cell_int[α] / L_α`` plus an
    optional sub-lattice offset.  The returned phase for site ``j`` is
    :math:`\\exp(2\\pi i\\,x_{j,\\alpha}/L_\\alpha)` (used as
    ``site_phases[i]`` when computing the many-body phase of a mask).

    Args:
        lattice: the real-space lattice.
        direction: **1-based** direction α of the polarization.
        include_sublattice: whether to add the sub-lattice offset to the
            crystal coordinate.

    Returns:
        np.ndarray: complex array of length ``n_site`` with the per-site
        phases.
    """
    if not 1 <= direction <= lattice.dim:
        raise ValueError(f"direction must be in 1..{lattice.dim}.")
    L = float(lattice.sample_size[direction - 1])
    phases = np.zeros(lattice.n_site, dtype=np.complex128)
    for (i, (cell_int, isub)) in enumerate(lattice.site_list):
        x = float(cell_int[direction - 1])
        if include_sublattice:
            x += float(lattice.sub_crys_list[isub - 1][direction - 1])
        phases[i] = cmath.exp(2j * math.pi * x / L)
    return phases


def _position_phase_for_mask(m: Mask, site_phases: np.ndarray) -> complex:
    """Many-body phase :math:`\\prod_{j \\in \\mathrm{occ}} e^{2\\pi i x_j/L}`.

    Args:
        m: occupation bitmask of the configuration.
        site_phases: per-site phases from
            :func:`many_body_position_phases`.

    Returns:
        complex: the product of the phases over the occupied sites.
    """
    z = COMPLEX_ONE
    tmp = m
    while tmp != 0:
        lsb = tmp & -tmp
        idx = (lsb.bit_length() - 1)  # 0-based site index
        z *= site_phases[idx]
        tmp ^= lsb
    return z


def _position_operator_matrix(
    basis_to: Symmetry_Sector_Basis,
    basis_from: Symmetry_Sector_Basis,
    site_phases: np.ndarray,
    particle_statistics: Particle_Statistics,
) -> sp.csr_matrix:
    """Projected position-operator matrix ``⟨χ_to|U_d|χ_from⟩``.

    Julia counterpart: ``_position_operator_matrix`` (sparse CSC; the
    Python port returns CSR).

    Args:
        basis_to: target symmetry-sector basis.
        basis_from: source symmetry-sector basis.
        site_phases: per-site phases of the position operator.
        particle_statistics: particle statistics of the model.

    Returns:
        scipy.sparse.csr_matrix: the projected position operator.
    """
    G = basis_from.symmetry_group
    assert basis_to.symmetry_group.name == G.name
    assert group_order(basis_to.symmetry_group) == group_order(G)
    assert basis_to.symmetry_group.n_site == G.n_site

    n_to = len(basis_to.representative_mask_list)
    n_from = len(basis_from.representative_mask_list)
    Is: list[int] = []
    Js: list[int] = []
    Vs: list[complex] = []

    chi_to = basis_to.irrep.values
    chi_from = basis_from.irrep.values
    inv_nG = 1.0 / group_order(G)

    for (col, repr_mask) in enumerate(basis_from.representative_mask_list):
        row = basis_to.representative_mask_to_mask_idx_map.get(repr_mask)
        if row is None:
            continue

        elem = 0.0 + 0.0j
        for (gidx, op) in enumerate(G.operations):
            shifted, _ = apply_operation_to_mask(
                repr_mask, op, particle_statistics
            )
            elem += (
                chi_to[gidx]
                * np.conj(chi_from[gidx])
                * _position_phase_for_mask(shifted, site_phases)
            )
        elem *= inv_nG

        if abs(elem) > 1e-13:
            Is.append(row)
            Js.append(col)
            Vs.append(elem)

    return sp.csr_matrix(
        (Vs, (Is, Js)), shape=(n_to, n_from)
    )


def _default_polarization_direction(dim: int, flux_direction: int) -> int:
    """Transverse direction to the flux (**1-based**).

    Args:
        dim: lattice dimension.
        flux_direction: **1-based** flux direction.

    Returns:
        int: **1-based** polarization direction
        (``flux_direction + 1`` wrapped, or the same in 1-D).
    """
    if dim == 1:
        return flux_direction
    return (flux_direction % dim) + 1


def _best_phase_permutation(raw: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Brute-force optimal assignment of branches to minimize discontinuity.

    Args:
        raw: current-step raw phases (length ``nbranch``).
        prev: previous-step tracked phases.

    Returns:
        np.ndarray: permutation of the branch indices.
    """
    nbranch = len(prev)
    used = [False] * nbranch
    current = [0] * nbranch
    best = list(range(nbranch))
    best_cost = [float("inf")]

    def visit(depth: int, cost: float) -> None:
        if depth > nbranch - 1:
            if cost < best_cost[0]:
                best_cost[0] = cost
                best[:] = current
            return
        for j in range(nbranch):
            if used[j]:
                continue
            delta = raw[j] - prev[depth]
            delta -= round(delta)
            next_cost = cost + delta**2
            if next_cost >= best_cost[0]:
                continue
            used[j] = True
            current[depth] = j
            visit(depth + 1, next_cost)
            used[j] = False

    visit(0, 0.0)
    return np.asarray(best, dtype=np.int64)


def _tracked_phase_branches(raw_phases: np.ndarray) -> np.ndarray:
    """Unwrap phase branches along the flux path.

    Args:
        raw_phases: ``(nθ, nbranch)`` matrix of raw branch phases.

    Returns:
        np.ndarray: the unwrapped (tracked) phases of the same shape.
    """
    nθ, nbranch = raw_phases.shape
    tracked = np.empty_like(raw_phases)
    order0 = np.argsort(raw_phases[0, :])
    tracked[0, :] = raw_phases[0, order0]
    prev = tracked[0, :].copy()

    for iθ in range(1, nθ):
        best_perm = _best_phase_permutation(raw_phases[iθ, :], prev)
        for b in range(nbranch):
            phi = raw_phases[iθ, best_perm[b]]
            tracked[iθ, b] = phi + round(prev[b] - phi)
        prev = tracked[iθ, :].copy()
    return tracked


@dataclass
class _ManifoldState:
    """Normalized ``(sector, level)`` specification of a manifold state.

    Attributes:
        sector: the irrep label tuple.
        level: the **1-based** eigenstate level within the sector.
    """

    sector: tuple
    level: int


def _normalize_manifold_state_specs(
    sector_labels, nev_per_sector: int, manifold_states
) -> tuple[list, list[_ManifoldState], dict, bool]:
    """Normalize a low-energy manifold to explicit ``(sector, level)`` specs.

    Args:
        sector_labels: ``"identity"`` or a list of sector label tuples.
        nev_per_sector: legacy one-size-fits-all number of levels per sector.
        manifold_states: optional explicit ``(sector, level)`` states.

    Returns:
        tuple: ``(labels, state_specs, required_levels, is_identity)``.
    """
    if nev_per_sector <= 0:
        raise ValueError("nev_per_sector must be positive.")
    is_identity = isinstance(sector_labels, str) and sector_labels == "identity"
    if is_identity:
        base_labels = ["identity"]
    elif isinstance(sector_labels, tuple):
        base_labels = [tuple(int(x) for x in sector_labels)]
    else:
        base_labels = [tuple(int(x) for x in l) for l in sector_labels]

    if manifold_states is None:
        specs = [
            _ManifoldState(sector=label, level=level)
            for label in base_labels
            for level in range(1, nev_per_sector + 1)
        ]
    else:
        if is_identity:
            raise ValueError(
                "Explicit manifold_states are only supported for "
                "symmetry-sector ED."
            )
        normalized: list[_ManifoldState] = []
        for state in manifold_states:
            if isinstance(state, dict) or hasattr(state, "sector"):
                sector_raw = state.sector if hasattr(state, "sector") else state["sector"]
                level_raw = state.level if hasattr(state, "level") else state["level"]
            else:
                sector_raw, level_raw = state[0], state[1]
            sector = tuple(int(x) for x in sector_raw)
            level = int(level_raw)
            if level <= 0:
                raise ValueError(
                    f"Manifold levels must be positive; got {level} in sector {sector}."
                )
            if sector not in base_labels:
                raise ValueError(
                    f"Manifold state ({sector}, level {level}) is absent from "
                    f"sector_labels={base_labels}."
                )
            normalized.append(_ManifoldState(sector=sector, level=level))
        if not normalized:
            raise ValueError("manifold_states must not be empty.")
        if len({(s.sector, s.level) for s in normalized}) != len(normalized):
            raise ValueError("manifold_states contains duplicate (sector, level) entries.")
        specs = normalized

    labels = list(dict.fromkeys(s.sector for s in specs))
    required_levels: dict = {}
    for state in specs:
        required_levels[state.sector] = max(
            required_levels.get(state.sector, 0), state.level
        )
    return labels, specs, required_levels, is_identity


def flux_charge_pump(
    model: Real_Space_Second_Quantized_Model,
    sector_labels,
    *,
    filling_fraction,
    flux_direction: int = 1,
    polarization_direction: int | None = None,
    twisted_phases_over_2π_list: list[float] | None = None,
    nev_per_sector: int = 1,
    manifold_states=None,
    include_sublattice: bool = True,
    checkpoint_dir: str = "checkpoints",
    fig_path: str | None = None,
    overwrite: bool = False,
) -> SimpleNamespace:
    """Compute the many-body charge pump under a twisted boundary condition.

    Julia counterpart: ``flux_charge_pump``.  Reads the full
    symmetry-resolved ED data from canonical per-θ checkpoints (produced by
    :func:`ed_scan` in flux-scan mode), projects Resta's periodic position
    operator :math:`\\exp(2\\pi i X/L)` into the low-energy manifold for the
    requested sectors, and unwraps the phase branches.

    Args:
        model: the second-quantized model.
        sector_labels: ``"identity"`` or a list of sector tuples.
        filling_fraction: particles per flattened vertex.
        flux_direction: **1-based** direction of the inserted flux.
        polarization_direction: **1-based** transverse direction of the
            polarization (default: transverse to the flux).
        twisted_phases_over_2π_list: flux values [in units of 2π] to scan;
            defaults to ``linspace(0, 1, 9)``.
        nev_per_sector: low-lying states per sector for the legacy
            one-size-fits-all manifold specification.
        manifold_states: optional explicit ``(sector, level)`` states;
            required when several manifold states occupy the same momentum
            sector.
        include_sublattice: whether the sub-lattice offset enters the
            polarization coordinate.
        checkpoint_dir: directory for the per-θ checkpoints.
        fig_path: optional path to save the pump plot.
        overwrite: recompute even if checkpoints exist.

    Returns:
        SimpleNamespace: ``pumped_charges``,
        ``pumped_charge_trajectories``, ``polarizations``,
        ``polarization_eigenvalues``, ``energies``, ``sector_labels``,
        ``flux_direction``, ``polarization_direction``,
        ``manifold_states``, ``is_identity``, ``fig_path``,
        ``checkpoint_paths``.
    """
    if twisted_phases_over_2π_list is None:
        twisted_phases_over_2π_list = list(np.linspace(0.0, 1.0, 9))

    labels, state_specs, required_levels, is_identity = (
        _normalize_manifold_state_specs(
            sector_labels, nev_per_sector, manifold_states
        )
    )

    dim = model.lattice.dim
    if not 1 <= flux_direction <= dim:
        raise ValueError(f"flux_direction must be in 1..{dim}.")
    if polarization_direction is None:
        polarization_direction = _default_polarization_direction(
            dim, flux_direction
        )
    if not 1 <= polarization_direction <= dim:
        raise ValueError(f"polarization_direction must be in 1..{dim}.")

    nstates = len(state_specs)
    max_level = max(required_levels.values())
    energies = np.full(
        (len(twisted_phases_over_2π_list), len(labels), max_level), np.nan
    )
    polarization_eigenvalues = np.zeros(
        (len(twisted_phases_over_2π_list), nstates), dtype=np.complex128
    )

    # ── Ensure all per-θ ED checkpoints exist (shared with spectrum flow) ──
    flux0 = [0.0] * model.lattice.dim
    from ..second_quantized_model import (
        update_second_quantized_model_with_twisted_phases,
    )

    update_second_quantized_model_with_twisted_phases(
        model, twisted_phases_over_2π=flux0
    )
    init_ed_data = build_ed_data(
        model,
        filling_fraction=filling_fraction,
        symmetry_group=build_translation_group(model.lattice, flux0),
    )

    checkpoint_paths = ed_scan(
        init_ed_data,
        nev=max(max_level, 2),
        mode="matrix",
        twisted_phases_over_2π_list=twisted_phases_over_2π_list,
        flux_direction=flux_direction,
        checkpoint_dir=checkpoint_dir,
        overwrite=overwrite,
        scanned_sectors=(None if is_identity else labels),
    )

    for iθ, ckpt_path in enumerate(checkpoint_paths):
        ed_data = load_checkpoint(ckpt_path)

        bases: dict = {}
        eigvecs: dict = {}
        for ilabel, label in enumerate(labels):
            irrep_idx = None
            for idx, irrep in enumerate(ed_data.irrep_list):
                if irrep.label == label:
                    irrep_idx = idx
                    break
            if irrep_idx is None or irrep_idx not in ed_data.ed_scan_res:
                print(f"[ED] Warning: sector {label} not found in checkpoint {ckpt_path}")
                continue
            vecs = ed_data.ed_scan_res[irrep_idx][1]
            nv = required_levels[label]
            if vecs.shape[1] < nv:
                raise ValueError(
                    f"Sector {label} in {ckpt_path} has {vecs.shape[1]} "
                    f"eigenvectors; need {nv}."
                )
            eigvecs[label] = vecs[:, :nv]
            vals = ed_data.ed_scan_res[irrep_idx][0]
            if len(vals) < nv:
                raise ValueError(
                    f"Sector {label} in {ckpt_path} has {len(vals)} "
                    f"eigenvalues; need {nv}."
                )
            energies[iθ, ilabel, :nv] = vals[:nv]

            irrep = ed_data.irrep_list[irrep_idx]
            bases[label] = build_symmetry_sector_basis(
                ed_data.orbit_catalog, irrep
            )

        # ── Position operator projected into the low-energy manifold ──
        site_phases = many_body_position_phases(
            model.lattice,
            polarization_direction,
            include_sublattice=include_sublattice,
        )
        P = np.zeros((nstates, nstates), dtype=np.complex128)

        position_blocks: dict = {}
        for (row, state_to) in enumerate(state_specs):
            for (col, state_from) in enumerate(state_specs):
                label_to, lev_to = state_to.sector, state_to.level
                label_from, lev_from = state_from.sector, state_from.level
                block_key = (label_to, label_from)
                if block_key not in position_blocks:
                    position_blocks[block_key] = _position_operator_matrix(
                        bases[label_to],
                        bases[label_from],
                        site_phases,
                        model.particle_statistics,
                    )
                Ublock = position_blocks[block_key]
                P[row, col] = np.vdot(
                    eigvecs[label_to][:, lev_to - 1],
                    Ublock @ eigvecs[label_from][:, lev_from - 1],
                )

        ev = np.linalg.eigvals(P)
        order = np.argsort(np.angle(ev))
        polarization_eigenvalues[iθ, :] = ev[order]

    # ── Unwrap phase branches and compute pumped charge ──
    raw_phases = np.angle(polarization_eigenvalues) / (2 * math.pi)
    polarizations = _tracked_phase_branches(raw_phases)
    pumped_charge_trajectories = polarizations - polarizations[0:1, :]
    pumped_charges = pumped_charge_trajectories[-1, :]

    # ── Plot ──
    if fig_path is not None:
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
        sectors_str = (
            "Full Hilbert Space"
            if is_identity
            else ", ".join(repr(l) for l in labels)
        )
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.set_xlabel(f"Inserted Flux [2π] along Direction-{flux_direction}")
        ax.set_ylabel("Pumped Charge ΔQ")
        ax.set_title(
            f"{model.lattice.sample_size}-sample Charge Pump — sectors: {sectors_str}"
        )
        for b in range(nstates):
            state = state_specs[b]
            lbl = (
                f"branch {b}"
                if is_identity
                else f"sector {state.sector!r}, level {state.level}"
            )
            ax.plot(
                twisted_phases_over_2π_list,
                pumped_charge_trajectories[:, b],
                linewidth=2,
                label=lbl,
            )
            ax.scatter(
                twisted_phases_over_2π_list,
                pumped_charge_trajectories[:, b],
                s=12,
            )
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(fig_path)
        print(f"[ED] charge pump plot saved to `{fig_path}`")

    print(
        f"[ED] Pumped charges for manifold states {state_specs} "
        f"{pumped_charges}"
    )

    return SimpleNamespace(
        twisted_phases_over_2π_list=twisted_phases_over_2π_list,
        energies=energies,
        sector_labels=labels,
        flux_direction=flux_direction,
        polarization_direction=polarization_direction,
        nev_per_sector=max_level,
        manifold_states=state_specs,
        polarization_eigenvalues=polarization_eigenvalues,
        polarizations=polarizations,
        pumped_charge_trajectories=pumped_charge_trajectories,
        pumped_charges=pumped_charges,
        include_sublattice=include_sublattice,
        is_identity=is_identity,
        fig_path=fig_path,
        checkpoint_paths=checkpoint_paths,
    )


__all__ = [
    "many_body_position_phases",
    "flux_charge_pump",
]
