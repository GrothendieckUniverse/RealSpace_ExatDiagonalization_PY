"""Spectrum flow under twisted boundary conditions.

Faithful port of ``observables/spectrum_flow.jl`` from the Julia package
``RealSpace_ExactDiagonalization.jl`` — a thin wrapper over the shared
per-flux ED checkpoints produced by :func:`ed_scan` in flux-scan mode.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..second_quantized_model import Real_Space_Second_Quantized_Model
from ..symmetry_resolved_ed import (
    build_ed_data,
    build_translation_group,
    ed_scan,
    load_checkpoint,
)


def flux_spectrum_flow(
    model: Real_Space_Second_Quantized_Model,
    sector_labels,
    *,
    filling_fraction,
    flux_direction: int = 1,
    twisted_phases_over_2π_list: list[float] | None = None,
    checkpoint_dir: str = "checkpoints",
    nev: int = 3,
    fig_path: str | None = None,
    overwrite: bool = False,
) -> SimpleNamespace:
    """Compute the spectrum flow under the twisted boundary condition.

    Julia counterpart: ``flux_spectrum_flow``.  Scans
    ``twisted_phases_over_2π_list`` along ``flux_direction``, reading the
    full symmetry-resolved ED data from canonical per-θ checkpoints
    produced by :func:`ed_scan` in flux-scan mode.  If a checkpoint is
    missing it is computed on the fly.

    Args:
        model: the second-quantized model.
        sector_labels: ``"identity"`` or a list of sector tuples, e.g.
            ``[(0, 0), (1, 0)]``.
        filling_fraction: particles per flattened vertex
            (:class:`fractions.Fraction` or int/float).
        flux_direction: **1-based** direction of the inserted flux.
        twisted_phases_over_2π_list: flux values [in units of 2π] to scan;
            defaults to ``linspace(0, 1, 9)``.
        checkpoint_dir: directory for the per-θ checkpoints.
        nev: eigenvalues per sector.
        fig_path: optional path to save the flow plot.
        overwrite: recompute even if checkpoints exist.

    Returns:
        SimpleNamespace: ``twisted_phases_over_2π_list``, ``energies``,
        ``sector_labels``, ``flux_direction``, ``nev``, ``is_identity``,
        ``fig_path``, ``checkpoint_paths``.
    """
    if twisted_phases_over_2π_list is None:
        twisted_phases_over_2π_list = list(np.linspace(0.0, 1.0, 9))

    # ── Resolve sector labels ──
    is_identity = False
    if isinstance(sector_labels, str):
        is_identity = sector_labels == "identity"
        labels = ["identity"] if is_identity else [sector_labels]
    elif isinstance(sector_labels, tuple):
        labels = [tuple(int(x) for x in sector_labels)]
    else:
        labels = [tuple(int(x) for x in l) for l in sector_labels]

    # ── Ensure all per-θ ED checkpoints exist ──
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
        nev=nev,
        mode="matrix",
        twisted_phases_over_2π_list=twisted_phases_over_2π_list,
        flux_direction=flux_direction,
        checkpoint_dir=checkpoint_dir,
        overwrite=overwrite,
        scanned_sectors=(None if is_identity else labels),
    )

    # ── Extract eigenvalues for requested sectors from each checkpoint ──
    energies = np.full(
        (len(twisted_phases_over_2π_list), len(labels), nev), np.nan
    )
    for iθ, ckpt_path in enumerate(checkpoint_paths):
        ed_data = load_checkpoint(ckpt_path)
        for isector, label in enumerate(labels):
            irrep_idx = None
            for idx, irrep in enumerate(ed_data.irrep_list):
                if irrep.label == label:
                    irrep_idx = idx
                    break
            if irrep_idx is not None and irrep_idx in ed_data.ed_scan_res:
                vals = ed_data.ed_scan_res[irrep_idx][0]
                nv = min(len(vals), nev)
                energies[iθ, isector, :nv] = vals[:nv]

    # ── Plot ──
    if fig_path is not None:
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
        sectors_str = (
            "Full Hilbert Space" if is_identity else ", ".join(repr(l) for l in labels)
        )
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.set_xlabel(f"Inserted Flux [2π] along Direction-{flux_direction}")
        ax.set_ylabel("E")
        ax.set_title(
            f"{model.lattice.sample_size}-sample Spectrum Flow — sectors: {sectors_str}"
        )
        for isector in range(len(labels)):
            for level in range(nev):
                lbl = (
                    f"sector {labels[isector]!r}"
                    if level == 0
                    else None
                )
                ax.plot(
                    twisted_phases_over_2π_list,
                    energies[:, isector, level],
                    alpha=1 - (level) / nev,
                    linewidth=2,
                    label=lbl,
                )
                ax.scatter(
                    twisted_phases_over_2π_list,
                    energies[:, isector, level],
                    s=12,
                    alpha=1 - (level) / nev,
                )
        if not is_identity:
            ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(fig_path)
        print(f"[ED] spectrum flow plot saved to `{fig_path}`")

    return SimpleNamespace(
        twisted_phases_over_2π_list=twisted_phases_over_2π_list,
        energies=energies,
        sector_labels=labels,
        flux_direction=flux_direction,
        nev=nev,
        is_identity=is_identity,
        fig_path=fig_path,
        checkpoint_paths=checkpoint_paths,
    )


__all__ = ["flux_spectrum_flow"]
