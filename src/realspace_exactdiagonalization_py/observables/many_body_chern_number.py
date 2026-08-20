"""Flux-torus Chern number of a selected many-body manifold.

Faithful port of ``observables/many_body_chern_number.jl`` from the Julia
package ``RealSpace_ExactDiagonalization.jl`` — the non-Abelian
Fukui–Hatsugai–Suzuki lattice Berry curvature for a selected low-energy
manifold over the two twisted-boundary fluxes.

The U(1) link variable is the phase of the determinant of the overlap matrix
between neighboring multiplet subspaces, and the returned ``chern_number`` is
``sum(plaquette phases) / 2π``.
"""

from __future__ import annotations

import os
import warnings
from fractions import Fraction
from types import SimpleNamespace

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
    build_translation_group,
    ed_scan,
)
from .entanglement_spectrum import _expand_sector_state_to_fock_amplitudes


def _normalize_flux_grid_size(flux_grid_size) -> tuple[int, int]:
    """Normalize a flux grid size to ``(nx, ny)``.

    Julia counterpart: ``_normalize_flux_grid_size``.

    Args:
        flux_grid_size: an integer (square grid) or a length-2 sequence.

    Returns:
        tuple[int, int]: ``(nx, ny)``.
    """
    if isinstance(flux_grid_size, int):
        nx = int(flux_grid_size)
        ny = int(flux_grid_size)
    else:
        nx = int(flux_grid_size[0])
        ny = int(flux_grid_size[1])
    if nx < 2 or ny < 2:
        raise ValueError("flux_grid_size must be at least 2 in both directions.")
    return nx, ny


def _chern_sector_labels(sector_labels) -> list:
    """Normalize Chern sector labels to a list of ``"identity"``/tuples.

    Julia counterpart: ``_chern_sector_labels``.

    Args:
        sector_labels: ``"identity"`` or a list of momentum tuples.

    Returns:
        list: the normalized labels.
    """
    if sector_labels == "identity":
        return ["identity"]
    return [tuple(int(x) for x in label) for label in sector_labels]


def _chern_params_digest(params: dict) -> str:
    """FNV-1a 64-bit hash of the (sorted) parameter pairs, as a hex string.

    Julia counterpart: ``_chern_params_digest``.  The FNV-1a offset basis and
    prime are preserved; the byte stream is ``repr`` of the sorted
    ``(key, value)`` pairs (Julia uses the ``repr`` of a vector of pairs, so
    the exact digest string differs from Julia's — it is used only as a
    checkpoint-filename tag).

    Args:
        params: the model parameter dict.

    Returns:
        str: hexadecimal digest.
    """
    pairs = sorted(params.items(), key=lambda kv: str(kv[0]))
    h = 0xCBF29CE484222325
    for b in repr(pairs).encode("utf-8"):
        h = ((h ^ b) * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return format(h, "x")


def _chern_flux_tag(flux) -> str:
    """Human-readable flux tag with ``.``/``-`` replaced by ``p``/``m``.

    Julia counterpart: ``_chern_flux_tag``.

    Args:
        flux: list of flux values.

    Returns:
        str: underscore-joined tag.
    """
    return "_".join(("%.6f" % θ).replace(".", "p").replace("-", "m") for θ in flux)


def _chern_checkpoint_path(
    model: Real_Space_Second_Quantized_Model,
    filling_fraction: Fraction,
    flux,
    checkpoint_dir: str,
) -> str:
    """Checkpoint filename for one flux point of the Chern scan.

    Julia counterpart: ``_chern_checkpoint_path`` (``.jld2``; the Python port
    uses ``.pkl`` to match the package's pickle-based checkpoints).

    Args:
        model: the second-quantized model.
        filling_fraction: particles per flattened vertex.
        flux: the flux vector.
        checkpoint_dir: checkpoint directory.

    Returns:
        str: the checkpoint path.
    """
    sample = "x".join(str(s) for s in model.lattice.sample_size)
    digest = _chern_params_digest(model.params)
    name = (
        f"many_body_chern_{model.tb_model.model_name}_{sample}_nu="
        f"{filling_fraction.numerator}_{filling_fraction.denominator}_"
        f"theta={_chern_flux_tag(flux)}_p={digest}.pkl"
    )
    return os.path.join(checkpoint_dir, name)


def _fock_amplitude_overlap(a: dict, b: dict) -> complex:
    """Overlap :math:`\\langle a | b\\rangle` of two Fock-amplitude maps.

    Julia counterpart: ``_fock_amplitude_overlap``.

    Args:
        a: ``mask → amplitude`` map.
        b: ``mask → amplitude`` map.

    Returns:
        complex: the overlap.
    """
    acc = 0.0 + 0.0j
    if len(a) <= len(b):
        for mask, amp in a.items():
            acc += np.conj(amp) * b.get(mask, 0.0 + 0.0j)
    else:
        for mask, amp in b.items():
            acc += np.conj(a.get(mask, 0.0 + 0.0j)) * amp
    return acc


def _subspace_overlap_matrix(states_a, states_b) -> np.ndarray:
    """Overlap matrix between two equal-dimension subspaces.

    Julia counterpart: ``_subspace_overlap_matrix``.

    Args:
        states_a: list of Fock-amplitude maps.
        states_b: list of Fock-amplitude maps.

    Returns:
        np.ndarray: complex ``(n, n)`` overlap matrix.
    """
    n = len(states_a)
    if len(states_b) != n:
        raise ValueError("Chern link compares subspaces of different dimensions.")
    M = np.empty((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            M[i, j] = _fock_amplitude_overlap(states_a[i], states_b[j])
    return M


def _unit_det_link(states_a, states_b, *, det_atol: float = 1e-10):
    """U(1) link = phase of the overlap-matrix determinant.

    Julia counterpart: ``_unit_det_link``.

    Args:
        states_a: list of Fock-amplitude maps.
        states_b: list of Fock-amplitude maps.
        det_atol: near-singular determinant threshold.

    Returns:
        tuple: ``(z/|z|, |z|)``; if ``|z| ≤ det_atol`` the link is replaced by
        ``1`` and a warning is emitted.
    """
    M = _subspace_overlap_matrix(states_a, states_b)
    z = np.linalg.det(M)
    if abs(z) <= det_atol:
        warnings.warn(
            "Near-singular Berry link; selected manifold may not be isolated.",
            stacklevel=2,
        )
        return 1.0 + 0.0j, abs(z)
    return z / abs(z), abs(z)


def _flux_point_states_for_chern(
    model: Real_Space_Second_Quantized_Model,
    labels,
    *,
    filling_fraction: Fraction,
    flux,
    nev_per_sector: int,
    mode: str,
    checkpoint_dir: str,
    overwrite: bool,
) -> SimpleNamespace:
    """Resolve the manifold Fock states at one flux point.

    Julia counterpart: ``_flux_point_states_for_chern``.

    Args:
        model: the second-quantized model.
        labels: normalized sector labels.
        filling_fraction: particles per flattened vertex.
        flux: the flux vector.
        nev_per_sector: states per sector.
        mode: ED mode.
        checkpoint_dir: checkpoint directory.
        overwrite: recompute even if a checkpoint exists.

    Returns:
        SimpleNamespace: ``states``, ``energies``, ``sector_level_labels``,
        ``checkpoint_path``.
    """
    update_second_quantized_model_with_twisted_phases(
        model, twisted_phases_over_2π=flux
    )
    active_group = (
        build_identity_group(model.lattice.n_site)
        if labels == ["identity"]
        else build_translation_group(model.lattice, flux)
    )
    ed_data = build_ed_data(
        model, filling_fraction=filling_fraction, symmetry_group=active_group
    )

    scanned = None if labels == ["identity"] else [label for label in labels]
    checkpoint_path = _chern_checkpoint_path(
        model, filling_fraction, flux, checkpoint_dir
    )
    ed_scan(
        ed_data,
        nev=nev_per_sector,
        mode=mode,
        scanned_sectors=scanned,
        checkpoint_path=checkpoint_path,
        overwrite=overwrite,
    )

    states: list[dict] = []
    energies: list[float] = []
    sector_level_labels: list[tuple] = []

    for label in labels:
        irrep_idx = None
        if label == "identity":
            irrep_idx = 0
        else:
            for idx, irrep in enumerate(ed_data.irrep_list):
                if irrep.label == label:
                    irrep_idx = idx
                    break
        if irrep_idx is None:
            raise ValueError(f"Sector {label!r} not found at flux {flux}.")
        if irrep_idx not in ed_data.ed_scan_res:
            raise ValueError(f"Sector {label!r} was not scanned at flux {flux}.")

        vals, vecs = ed_data.ed_scan_res[irrep_idx]
        n_take = min(nev_per_sector, vecs.shape[1])
        if n_take != nev_per_sector:
            raise ValueError(
                f"Requested {nev_per_sector} states in sector {label!r}, "
                f"only found {n_take}."
            )
        basis = build_symmetry_sector_basis(
            ed_data.orbit_catalog, ed_data.irrep_list[irrep_idx]
        )

        for level in range(n_take):
            states.append(
                _expand_sector_state_to_fock_amplitudes(
                    vecs[:, level], basis, model.particle_statistics
                )
            )
            energies.append(float(vals[level]))
            sector_level_labels.append((label, level + 1))

    return SimpleNamespace(
        states=states,
        energies=energies,
        sector_level_labels=sector_level_labels,
        checkpoint_path=checkpoint_path,
    )


def _plot_many_body_chern_curvature(result, *, fig_path: str | None = None):
    """Plot the flux-torus Berry curvature.

    Julia counterpart: ``_plot_many_body_chern_curvature`` (CairoMakie
    ``:balance`` colormap → matplotlib ``RdBu``).

    Args:
        result: the result of :func:`many_body_chern_number`.
        fig_path: optional path to save the figure.

    Returns:
        tuple: ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.set_xlabel("θ_x / 2π")
    ax.set_ylabel("θ_y / 2π")
    ax.set_title(f"Flux-torus Berry curvature, C={result.chern_number:.6f}")
    hm = ax.pcolormesh(
        result.flux_x, result.flux_y, result.berry_curvature.T,
        shading="auto", cmap="RdBu",
    )
    fig.colorbar(hm, ax=ax, label="plaquette phase")

    if fig_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
        fig.savefig(fig_path)
    return fig, ax


def many_body_chern_number(
    model: Real_Space_Second_Quantized_Model,
    sector_labels,
    *,
    filling_fraction: Fraction,
    flux_grid_size=(5, 5),
    nev_per_sector: int = 1,
    mode: str = "matrix",
    checkpoint_dir: str = "checkpoints",
    overwrite: bool = False,
    fig_path: str | None = None,
    det_atol: float = 1e-10,
) -> SimpleNamespace:
    """Chern number of a many-body manifold on the twisted-boundary flux torus.

    Julia counterpart: ``many_body_chern_number``.

    Args:
        model: the second-quantized model (must be 2-D).
        sector_labels: ``"identity"`` or the set of momentum sectors whose
            lowest ``nev_per_sector`` states define the manifold at each flux.
        filling_fraction: particles per flattened vertex.
        flux_grid_size: ``(nx, ny)`` flux grid (or a single integer).
        nev_per_sector: states per sector.
        mode: ED mode.
        checkpoint_dir: checkpoint directory.
        overwrite: recompute even if checkpoints exist.
        fig_path: optional path to save the curvature plot.
        det_atol: near-singular determinant threshold.

    Returns:
        SimpleNamespace: ``chern_number``, ``rounded_chern``,
        ``berry_curvature``, ``flux_x``, ``flux_y``, ``sector_labels``,
        ``nev_per_sector``, ``flux_grid_size``, ``energies``,
        ``min_link_det``, ``link_det_abs_x``, ``link_det_abs_y``,
        ``checkpoint_paths``, ``fig_path``.
    """
    filling_fraction = Fraction(filling_fraction)
    if model.lattice.dim != 2:
        raise ValueError("many_body_chern_number currently expects a 2D lattice.")
    nx, ny = _normalize_flux_grid_size(flux_grid_size)
    labels = _chern_sector_labels(sector_labels)

    flux_x = np.arange(nx) / nx
    flux_y = np.arange(ny) / ny
    states = [[None] * ny for _ in range(nx)]
    energies = [[None] * ny for _ in range(nx)]
    checkpoint_paths = [[None] * ny for _ in range(nx)]

    os.makedirs(checkpoint_dir, exist_ok=True)
    for ix in range(nx):
        for iy in range(ny):
            flux = [float(flux_x[ix]), float(flux_y[iy])]
            print(f"[Chern] flux point ({ix}/{nx}, {iy}/{ny}) flux={flux} labels={labels}")
            point = _flux_point_states_for_chern(
                model,
                labels,
                filling_fraction=filling_fraction,
                flux=flux,
                nev_per_sector=nev_per_sector,
                mode=mode,
                checkpoint_dir=checkpoint_dir,
                overwrite=overwrite,
            )
            states[ix][iy] = point.states
            energies[ix][iy] = point.energies
            checkpoint_paths[ix][iy] = point.checkpoint_path

    Ux = np.empty((nx, ny), dtype=np.complex128)
    Uy = np.empty((nx, ny), dtype=np.complex128)
    det_abs_x = np.empty((nx, ny), dtype=np.float64)
    det_abs_y = np.empty((nx, ny), dtype=np.float64)
    for ix in range(nx):
        for iy in range(ny):
            ixp = (ix + 1) % nx
            iyp = (iy + 1) % ny
            Ux[ix, iy], det_abs_x[ix, iy] = _unit_det_link(
                states[ix][iy], states[ixp][iy], det_atol=det_atol
            )
            Uy[ix, iy], det_abs_y[ix, iy] = _unit_det_link(
                states[ix][iy], states[ix][iyp], det_atol=det_atol
            )

    berry_curvature = np.empty((nx, ny), dtype=np.float64)
    for ix in range(nx):
        for iy in range(ny):
            ixp = (ix + 1) % nx
            iyp = (iy + 1) % ny
            berry_curvature[ix, iy] = float(
                np.angle(
                    Ux[ix, iy]
                    * Uy[ixp, iy]
                    * np.conj(Ux[ix, iyp])
                    * np.conj(Uy[ix, iy])
                )
            )

    chern_number = float(np.sum(berry_curvature)) / (2 * np.pi)
    res = SimpleNamespace(
        chern_number=chern_number,
        rounded_chern=round(chern_number),
        berry_curvature=berry_curvature,
        flux_x=flux_x,
        flux_y=flux_y,
        sector_labels=labels,
        nev_per_sector=nev_per_sector,
        flux_grid_size=(nx, ny),
        energies=energies,
        min_link_det=min(float(det_abs_x.min()), float(det_abs_y.min())),
        link_det_abs_x=det_abs_x,
        link_det_abs_y=det_abs_y,
        checkpoint_paths=checkpoint_paths,
        fig_path=fig_path,
    )

    if fig_path is not None:
        _plot_many_body_chern_curvature(res, fig_path=fig_path)
    return res


__all__ = ["many_body_chern_number"]
