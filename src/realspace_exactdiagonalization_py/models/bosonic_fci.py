"""Bosonic FCI on the Haldane honeycomb lattice.

Faithful port of ``test/bosonic_fci.jl`` from the Julia package
``RealSpace_ExactDiagonalization.jl``: the extended Bose–Hubbard model on
the Haldane honeycomb lattice at half filling of the lower Chern band
(D. N. Sheng, Z.-C. Gu, K. Sun, L. Sheng, *Phys. Rev. Lett.* **107**,
146803 (2011)).
"""

from __future__ import annotations

import cmath
import math
from fractions import Fraction
from typing import Any

import numpy as np

from tightbinding_py import (
    add_hopping_term,
    generate_bilinear_terms,
    initialize_real_space_lattice,
    initialize_real_space_tightbinding_model,
)

from ..second_quantized_model import (
    Particle_Statistics,
    Real_Space_Second_Quantized_Model,
)
from ..symmetry_resolved_ed import build_ed_data, build_translation_group, ed_scan

#: Parameters from Wang, Gu, Gong, and Sheng's bosonic FCI
#: [PhysRevLett.107.146803].  Julia key ``t′`` → Python key ``"t'"``;
#: ``t′′`` → ``"t''"``; ``ϕ_over_2π`` → ``"phi_over_2pi"``.
params_DNSheng: dict[str, float] = {
    "t": 1.0,          # nearest-neighbor hopping
    "t'": 0.60,        # next-nearest-neighbor hopping
    "t''": -0.58,      # next-next-nearest-neighbor hopping
    "phi_over_2pi": 0.2,  # flux 0.4π
    "V1": 0.0,         # nearest-neighbor density-density interaction
    "V2": 0.0,         # next-nearest-neighbor density-density interaction
}


def build_zero_flux_bosonic_fci_second_quantized_model(
    *,
    sample_size: list[int] | None = None,
    params: dict[str, Any] | None = None,
) -> Real_Space_Second_Quantized_Model:
    """Second-quantized model for the bosonic FCI on Haldane honeycomb.

    Args:
        sample_size: the sample size along each primitive lattice vector
            (default ``[3, 4]``).
        params: model parameters using D. N. Sheng's parameters (default
            :data:`params_DNSheng`).

    Returns:
        Real_Space_Second_Quantized_Model: the assembled model with zero
        twisted phases.
    """
    if sample_size is None:
        sample_size = [3, 4]
    if params is None:
        params = dict(params_DNSheng)
    t = params["t"]
    t1 = params["t'"]
    t2 = params["t''"]
    phi_over_2pi = params["phi_over_2pi"]
    V1 = params["V1"]
    V2 = params["V2"]

    lattice = initialize_real_space_lattice(
        sample_size=sample_size,
        brav_vec_list=[[1.0, 0.0], [1 / 2, math.sqrt(3) / 2]],
        sub_crys_list=[[0.0, 0.0], [1 / 3, 1 / 3]],
        lattice_name="Haldane_Honeycomb",
        pbc_indicator=[True, True],
    )
    tb_model = initialize_real_space_tightbinding_model(
        lattice, model_name="haldane_boson_FCI"
    )

    phi = 2 * math.pi * phi_over_2pi

    # Nearest-neighbor (inter-sublattice, real)
    add_hopping_term(tb_model, ((((0, 0), 1), ((0, 0), 2)), -t))
    add_hopping_term(tb_model, ((((0, 0), 1), ((0, -1), 2)), -t))
    add_hopping_term(tb_model, ((((0, 0), 1), ((-1, 0), 2)), -t))

    # NNN (intra-sublattice, complex — Haldane flux)
    add_hopping_term(
        tb_model, ((((0, 0), 1), ((1, 0), 1)), -t1 * cmath.exp(1j * phi))
    )
    add_hopping_term(
        tb_model, ((((0, 0), 1), ((0, 1), 1)), -t1 * cmath.exp(-1j * phi))
    )
    add_hopping_term(
        tb_model, ((((0, 0), 1), ((-1, 1), 1)), -t1 * cmath.exp(1j * phi))
    )
    add_hopping_term(
        tb_model, ((((0, 0), 2), ((1, 0), 2)), -t1 * cmath.exp(-1j * phi))
    )
    add_hopping_term(
        tb_model, ((((0, 0), 2), ((0, 1), 2)), -t1 * cmath.exp(1j * phi))
    )
    add_hopping_term(
        tb_model, ((((0, 0), 2), ((-1, 1), 2)), -t1 * cmath.exp(-1j * phi))
    )

    # Third-nearest-neighbor (inter-sublattice, real)
    add_hopping_term(tb_model, ((((0, 0), 2), ((1, 1), 1)), -t2))
    add_hopping_term(tb_model, ((((1, 0), 1), ((0, 1), 2)), -t2))
    add_hopping_term(tb_model, ((((1, 0), 2), ((0, 1), 1)), -t2))

    # Build the model with initialized ZERO twisted phases
    bilinear_terms = generate_bilinear_terms(
        tb_model, twisted_phases_over_2π=[0.0] * lattice.dim
    )
    density_terms: list[tuple[int, int, complex]] = []

    added_density_pairs: set[tuple[int, int]] = set()

    def add_density_pair(i: int, j: int, V: float) -> None:
        if i == j:
            return
        pair = tuple(sorted((i, j)))
        if pair in added_density_pairs:
            return
        added_density_pairs.add(pair)
        density_terms.append((pair[0], pair[1], complex(V)))

    # NN density-density interaction: the same three inter-sublattice bonds
    # as the NN hopping geometry, counted once as unordered pairs.
    if V1 != 0.0:
        for (cell, isub) in lattice.site_list:
            if isub != 1:
                continue
            i = lattice.site_to_index_map[(cell, 1)]
            for shift in ([0, 0], [0, -1], [-1, 0]):
                cell_to = tuple(
                    (c + s) % L
                    for c, s, L in zip(cell, shift, lattice.sample_size)
                )
                j = lattice.site_to_index_map[(cell_to, 2)]
                add_density_pair(i, j, V1)

    # NNN density-density interaction: the same three intra-sublattice
    # directions as the Haldane NNN hopping geometry.
    if V2 != 0.0:
        for (cell, isub) in lattice.site_list:
            for shift in ([1, 0], [0, 1], [-1, 1]):
                cell_to = tuple(
                    (c + s) % L
                    for c, s, L in zip(cell, shift, lattice.sample_size)
                )
                j = lattice.site_to_index_map[(cell_to, isub)]
                i = lattice.site_to_index_map[(cell, isub)]
                add_density_pair(i, j, V2)

    return Real_Space_Second_Quantized_Model(
        params=dict(params),
        lattice=lattice,
        tb_model=tb_model,
        particle_statistics=Particle_Statistics.BOSONIC,
        bilinear_terms=bilinear_terms,
        density_density_terms=density_terms,
    )


def default_fci_sectors(sample_size: list[int]) -> list[tuple[int, int]]:
    """Crystal-momentum sector labels hosting the two bosonic semion FCI states.

    Verified via a full ED scan (see the Julia ``test/bosonic_fci.jl``).

    Args:
        sample_size: the sample size.

    Returns:
        list[tuple[int, int]]: the momentum labels of the nearly-degenerate
        FCI ground states.
    """
    if sample_size == [2, 3]:
        return [(0, 0), (1, 0)]
    elif sample_size == [3, 2]:
        return [(0, 0), (0, 1)]
    elif sample_size == [3, 4]:
        return [(0, 0), (0, 2)]
    elif sample_size == [4, 3]:
        return [(0, 0), (2, 0)]
    print(
        f"WARNING: Unknown `sample_size={sample_size}` — defaulting sector "
        "labels to [(0,0)]. The user should manually specify the FCI "
        "ground-state sectors."
    )
    return [(0, 0)]


def test_bosonic_fci_spectrum_flow(
    *,
    sample_size: list[int] | None = None,
    params: dict[str, Any] | None = None,
    filling_fraction: Fraction | int | float = Fraction(1, 2),  # per flatband
    mode: str = "sectors",
    flux_direction: int | None = None,
    twisted_phases_over_2π_list: list[float] | None = None,
):
    """Spectrum flow for the bosonic FCI on Haldane honeycomb.

    Args:
        sample_size: sample size (default ``[2, 3]``).
        params: model parameters (default :data:`params_DNSheng`).
        filling_fraction: filling **per flatband** (default 1/2).  Note:
            the ``filling_fraction`` input of the ED constructor is the
            fractional filling per flattened vertex (= half of this).
        mode: ``"sectors"`` for symmetry-resolved sectors, ``"identity"``
            for the full Hilbert space.
        flux_direction: **1-based** flux direction; defaults to the first
            direction whose sample size is divisible by the GSD of the
            topological state (2 for the semion TO here).
        twisted_phases_over_2π_list: twisted phases to scan
            (default ``linspace(0, 1, 9)``).

    Returns:
        SimpleNamespace: the result of :func:`flux_spectrum_flow`.
    """
    if sample_size is None:
        sample_size = [2, 3]
    if params is None:
        params = dict(params_DNSheng)
    if twisted_phases_over_2π_list is None:
        twisted_phases_over_2π_list = list(np.linspace(0.0, 1.0, 9))
    if flux_direction is None:
        flux_direction = next(
            (d for d in range(1, len(sample_size) + 1) if sample_size[d - 1] % 2 == 0),
            1,
        )

    print(
        f"[ED] Twisted phase list to be computed: {twisted_phases_over_2π_list}"
    )
    model = build_zero_flux_bosonic_fci_second_quantized_model(
        sample_size=sample_size, params=params
    )
    print(f"[ED] The chosen twisted phase direction: {flux_direction}")

    lattice = model.lattice
    filling_fraction_vertex = Fraction(filling_fraction) / 2
    n_filled = int(lattice.n_site * filling_fraction_vertex)
    print(
        f"[ED] Bosonic FCI over {sample_size[0]}×{sample_size[1]} Haldane "
        f"honeycomb lattice with {lattice.n_site} sites, {n_filled} bosons "
        f"(vertices filling {filling_fraction_vertex})"
    )

    if mode not in ("identity", "sectors"):
        raise ValueError("mode must be 'identity' or 'sectors'.")
    is_identity = mode == "identity"
    labels: Any = "identity" if is_identity else default_fci_sectors(sample_size)
    tag = "identity" if is_identity else "sectors"

    import os

    from ..observables.spectrum_flow import flux_spectrum_flow

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    result = flux_spectrum_flow(
        model,
        labels,
        filling_fraction=filling_fraction_vertex,
        flux_direction=flux_direction,
        twisted_phases_over_2π_list=twisted_phases_over_2π_list,
        nev=5,
        fig_path=os.path.join(
            project_root,
            "figures",
            f"bosonic_FCI_spectrum_flow_{tag}_{sample_size}.svg",
        ),
        checkpoint_dir=os.path.join(project_root, "checkpoints"),
    )
    return result


def test_bosonic_fci_charge_pump(
    *,
    sample_size: list[int] | None = None,
    params: dict[str, Any] | None = None,
    filling_fraction: Fraction | int | float = Fraction(1, 2),  # per flatband
    mode: str = "sectors",
    flux_direction: int | None = None,
    polarization_direction: int | None = None,
    twisted_phases_over_2π_list: list[float] | None = None,
    atol: float = 0.08,
):
    """Fractional charge pump for the bosonic FCI on Haldane honeycomb.

    For the half-filled Chern band, one flux quantum should advance each
    polarization branch by approximately one half charge — the finite-size
    version of the bosonic ν = 1/2 Laughlin pump.

    Args:
        sample_size: sample size (default ``[2, 3]``).
        params: model parameters (default :data:`params_DNSheng`).
        filling_fraction: filling **per flatband**.
        mode: ``"sectors"`` or ``"identity"``.
        flux_direction: **1-based** flux direction (default: first
            even-size direction, for the semion GSD = 2).
        polarization_direction: **1-based** transverse polarization
            direction (default: transverse to the flux).
        twisted_phases_over_2π_list: flux values to scan.
        atol: tolerance for the charge-quantization check.

    Returns:
        SimpleNamespace: the result of :func:`flux_charge_pump`.
    """
    if sample_size is None:
        sample_size = [2, 3]
    if params is None:
        params = dict(params_DNSheng)
    if twisted_phases_over_2π_list is None:
        twisted_phases_over_2π_list = list(np.linspace(0.0, 1.0, 9))
    if flux_direction is None:
        flux_direction = next(
            (d for d in range(1, len(sample_size) + 1) if sample_size[d - 1] % 2 == 0),
            1,
        )
    if polarization_direction is None:
        from ..observables.charge_pump import _default_polarization_direction

        polarization_direction = _default_polarization_direction(
            len(sample_size), flux_direction
        )

    model = build_zero_flux_bosonic_fci_second_quantized_model(
        sample_size=sample_size, params=params
    )
    lattice = model.lattice
    filling_fraction_vertex = Fraction(filling_fraction) / 2
    n_filled = int(lattice.n_site * filling_fraction_vertex)
    print(
        f"[ED] Bosonic FCI over {sample_size[0]}×{sample_size[1]} Haldane "
        f"honeycomb lattice with {lattice.n_site} sites, {n_filled} bosons "
        f"(vertices filling {filling_fraction_vertex})"
    )

    if mode not in ("identity", "sectors"):
        raise ValueError("mode must be 'identity' or 'sectors'.")
    is_identity = mode == "identity"
    labels: Any = "identity" if is_identity else default_fci_sectors(sample_size)
    tag = "identity" if is_identity else "sectors"

    import os

    from ..observables.charge_pump import flux_charge_pump

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    result = flux_charge_pump(
        model,
        labels,
        filling_fraction=filling_fraction_vertex,
        flux_direction=flux_direction,
        polarization_direction=polarization_direction,
        twisted_phases_over_2π_list=twisted_phases_over_2π_list,
        nev_per_sector=1,
        fig_path=os.path.join(
            project_root,
            "figures",
            f"bosonic_FCI_charge_pump_{tag}_{sample_size}.svg",
        ),
        checkpoint_dir=os.path.join(project_root, "checkpoints"),
    )

    if (
        mode == "sectors"
        and abs(twisted_phases_over_2π_list[-1] - 1.0) < 1e-12
    ):
        pumped = np.abs(result.pumped_charges)
        print(f"[ED] Pumped charges |ΔQ| = {pumped}")
        print(
            "[ED] "
            + (
                "PASS: all branches pumped ≈ 0.5 charge."
                if np.all(np.abs(pumped - 0.5) < atol)
                else "FAIL: a branch deviates from 0.5 by more than atol."
            )
        )
    return result


def build_bosonic_fci_ed_data(
    *,
    sample_size: list[int] = None,
    params: dict[str, Any] = None,
    filling_fraction_per_band: Fraction | int | float = Fraction(1, 2),
):
    """Build the symmetry-resolved ED data for the bosonic Haldane FCI.

    Convenience wrapper used by the benchmark and the examples.

    Args:
        sample_size: sample size (default ``[3, 4]``).
        params: model parameters (default :data:`params_DNSheng`).
        filling_fraction_per_band: filling per flatband (default 1/2);
            the vertex filling is half of this.

    Returns:
        Symmetry_Resolved_ED_Data: ED data with the translation group.
    """
    if sample_size is None:
        sample_size = [3, 4]
    if params is None:
        params = dict(params_DNSheng)
    model = build_zero_flux_bosonic_fci_second_quantized_model(
        sample_size=sample_size, params=params
    )
    filling_fraction = Fraction(filling_fraction_per_band) / model.lattice.n_sub
    symmetry_group = build_translation_group(model.lattice)
    return build_ed_data(
        model, filling_fraction=filling_fraction, symmetry_group=symmetry_group
    )


__all__ = [
    "params_DNSheng",
    "build_zero_flux_bosonic_fci_second_quantized_model",
    "build_bosonic_fci_ed_data",
    "default_fci_sectors",
    "test_bosonic_fci_spectrum_flow",
    "test_bosonic_fci_charge_pump",
]
