"""Fermionic FCI on the checkerboard lattice.

Faithful port of ``test/fermionic_fci.jl`` from the Julia package
``RealSpace_ExactDiagonalization.jl``: the spinless-fermion model on the
checkerboard lattice with two flat bands of Chern numbers ±1
(K. Sun, Z. Gu, H. Katsura, S. Das Sarma, "Nearly Flatbands with Nontrivial
Topology", arXiv:1012.5864).  At ν = 2/3 filling of the lower band the
interacting ground state is a fermionic fractional Chern insulator with
three nearly-degenerate ground states on the torus (GSD = 3).
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
from ..symmetry_resolved_ed import (
    build_ed_data,
    build_translation_group,
    ed_scan,
    print_spectrum,
)

#: Parameters from Sun, Gu, Katsura, and Sarma [arXiv:1012.5864].
#: Julia key ``t′_1`` → Python key ``"t'_1"``, ``ϕ_over_2π`` →
#: ``"phi_over_2pi"``.
params_Sun_Gu_Katsura_Sarma: dict[str, float] = {
    "t": 1.0,                          # nearest-neighbor hopping
    "t'_1": 1 / (2 + math.sqrt(2)),    # next-nearest-neighbor hopping kind 1
    "t'_2": -1 / (2 + math.sqrt(2)),   # next-nearest-neighbor hopping kind 2
    "t''": 1 / (2 + 2 * math.sqrt(2)),  # next-next-nearest-neighbor hopping
    "phi_over_2pi": 1 / 8,             # flux per 2π (time-reversal breaking)
    "V1": 2.0,   # NN density-density interaction (inter-sublattice)
    "V2": 1.0,   # NNN interaction (same sublattice, x/y)
    "V3": 0.0,   # NNNN interaction (same sublattice, diagonal)
}

#: Denominator of the dimensionless t′′ numerator convention of the phase
#: exploration: ``params["t''"] = t_value / (2 + 2√2)``.
CHECKERBOARD_TPP_DENOMINATOR = 2 + 2 * math.sqrt(2)


def build_fermionic_checkerboard_tb_model(
    *,
    sample_size: list[int] | None = None,
    params: dict[str, Any] | None = None,
    flip_bands: bool = True,
):
    """Tight-binding model for the checkerboard lattice with staggered flux.

    Args:
        sample_size: sample size (default ``[3, 4]``).
        params: model parameters (default
            :data:`params_Sun_Gu_Katsura_Sarma`).
        flip_bands: flip the sign of all hoppings to make the lower band
            flat.

    Returns:
        Real_Space_TightBinding_Model: the checkerboard tight-binding model.
    """
    if sample_size is None:
        sample_size = [3, 4]
    if params is None:
        params = dict(params_Sun_Gu_Katsura_Sarma)
    p = dict(params)
    r_data = initialize_real_space_lattice(
        lattice_name="checkerboard",
        sample_size=sample_size,
        brav_vec_list=[[1.0, 0.0], [0.0, 1.0]],
        sub_crys_list=[[0.5, 0], [0.0, 0.5]],
        pbc_indicator=[True, True],
    )
    tb_model = initialize_real_space_tightbinding_model(
        r_data, model_name="checkerboard"
    )

    if flip_bands:
        p["t"] = -p["t"]
        p["t'_1"] = -p["t'_1"]
        p["t'_2"] = -p["t'_2"]
        p["t''"] = -p["t''"]

    t, t1_1, t1_2, t2 = p["t"], p["t'_1"], p["t'_2"], p["t''"]
    phi = 2 * math.pi * p["phi_over_2pi"]

    # NN hoppings (inter-sublattice, complex — staggered flux)
    add_hopping_term(
        tb_model, ((((0, 0), 1), ((0, 0), 2)), -t * cmath.exp(-1j * phi))
    )
    add_hopping_term(
        tb_model, ((((0, 0), 1), ((1, 0), 2)), -t * cmath.exp(1j * phi))
    )
    add_hopping_term(
        tb_model, ((((0, 0), 2), ((0, 1), 1)), -t * cmath.exp(-1j * phi))
    )
    add_hopping_term(
        tb_model, ((((1, 0), 2), ((0, 1), 1)), -t * cmath.exp(1j * phi))
    )

    # NNN hoppings (intra-sublattice, real, anisotropic)
    add_hopping_term(tb_model, ((((0, 0), 1), ((1, 0), 1)), -t1_1))
    add_hopping_term(tb_model, ((((0, 0), 1), ((0, 1), 1)), -t1_2))
    add_hopping_term(tb_model, ((((0, 0), 2), ((1, 0), 2)), -t1_2))
    add_hopping_term(tb_model, ((((0, 0), 2), ((0, 1), 2)), -t1_1))

    # NNNN hoppings (intra-sublattice, diagonal, real)
    add_hopping_term(tb_model, ((((0, 0), 1), ((1, 1), 1)), -t2))
    add_hopping_term(tb_model, ((((0, 0), 2), ((1, 1), 2)), -t2))
    add_hopping_term(tb_model, ((((1, 0), 2), ((0, 1), 2)), -t2))
    add_hopping_term(tb_model, ((((0, 1), 1), ((1, 0), 1)), -t2))

    print(
        f"[ED] Checkerboard TB model: {sample_size} unit cells × "
        f"{r_data.n_sub} sublattices = {r_data.n_site} sites"
    )
    return tb_model


def build_zero_flux_fermionic_fci_second_quantized_model(
    *,
    sample_size: list[int] | None = None,
    params: dict[str, Any] | None = None,
) -> Real_Space_Second_Quantized_Model:
    """Second-quantized model for the fermionic FCI on checkerboard.

    The model has two flat bands with Chern numbers ±1.  At ν = 2/3
    filling of the lower band the interacting ground state is a fermionic
    fractional Chern insulator with three nearly-degenerate ground states
    on the torus (GSD = 3).

    Args:
        sample_size: sample size (default ``[3, 4]``).
        params: model parameters (default
            :data:`params_Sun_Gu_Katsura_Sarma`).

    Returns:
        Real_Space_Second_Quantized_Model: the assembled model with zero
        twisted phases.
    """
    if sample_size is None:
        sample_size = [3, 4]
    if params is None:
        params = dict(params_Sun_Gu_Katsura_Sarma)
    tb_model = build_fermionic_checkerboard_tb_model(
        sample_size=sample_size, params=params
    )
    lattice = tb_model.lattice
    p = dict(params)

    bilinear_terms = generate_bilinear_terms(
        tb_model, twisted_phases_over_2π=[0.0] * lattice.dim
    )

    V1, V2, V3 = p["V1"], p["V2"], p["V3"]
    L1, L2 = lattice.sample_size
    density_terms: list[tuple[int, int, complex]] = []

    for (i_from, (cell_from, sub_from)) in enumerate(lattice.site_list, start=1):
        for (i_to, (cell_to, sub_to)) in enumerate(lattice.site_list, start=1):
            if i_from == i_to:
                continue
            cell_shift = tuple(
                (c2 - c1) % L
                for c1, c2, L in zip(cell_from, cell_to, lattice.sample_size)
            )

            # V1: NN (inter-sublattice, 4 bonds)
            if (
                V1 != 0.0
                and sub_from == 1
                and sub_to == 2
                and (
                    cell_shift == (0, 0)
                    or cell_shift == (1, 0)
                    or cell_shift == (0, L2 - 1)
                    or cell_shift == (1, L2 - 1)
                )
            ):
                density_terms.append((i_from, i_to, complex(V1)))

            # V2: NNN (same sublattice, x or y)
            if (
                V2 != 0.0
                and sub_from == sub_to
                and (cell_shift == (1, 0) or cell_shift == (0, 1))
            ):
                density_terms.append((i_from, i_to, complex(V2)))

            # V3: NNNN (same sublattice, diagonal)
            if (
                V3 != 0.0
                and sub_from == sub_to
                and (cell_shift == (1, 1) or cell_shift == (1, L2 - 1))
            ):
                density_terms.append((i_from, i_to, complex(V3)))

    return Real_Space_Second_Quantized_Model(
        params=p,
        lattice=lattice,
        tb_model=tb_model,
        particle_statistics=Particle_Statistics.FERMIONIC,
        bilinear_terms=bilinear_terms,
        density_density_terms=density_terms,
    )


def default_fci_sectors_fermionic(
    sample_size: list[int],
) -> list[tuple[int, int]]:
    """Momentum sector labels hosting the fermionic FCI ground states.

    For the ν = 2/3 fermionic FCI on checkerboard, three nearly-degenerate
    ground states (GSD = 3) occupy the k_y = 0 sectors.  Verified via full
    ED scan on [3,4] with V₁ = 2.0, V₂ = 1.0:

        E₁ ≈ E₂ ≈ E₃  at k = (0,0), (1,0), (2,0).

    The three states are split by finite-size effects of order ∼ 0.05–0.10 t.

    Args:
        sample_size: the sample size.

    Returns:
        list[tuple[int, int]]: the FCI ground-state momentum labels.

    Raises:
        ValueError: for unknown sample sizes.
    """
    if sample_size == [2, 3]:
        return [(0, 0), (0, 1), (0, 2)]
    elif sample_size == [3, 2]:
        return [(0, 0), (1, 0), (2, 0)]
    elif sample_size == [3, 4]:
        return [(0, 0), (1, 0), (2, 0)]
    elif sample_size == [4, 3]:
        return [(0, 0), (0, 1), (0, 2)]
    raise ValueError(
        f"Unknown sample_size={sample_size} — run a full ED scan to "
        "identify the FCI sectors."
    )


def test_fermionic_fci_full_ed(
    *,
    sample_size: list[int] | None = None,
    params: dict[str, Any] | None = None,
    filling_fraction: Fraction | int | float = Fraction(1, 3),
    nev: int = 5,
):
    """Full symmetry-resolved ED scan of the checkerboard FCI model.

    Reports the lowest eigenvalues by momentum sector — the first
    diagnostic step to identify the nearly-degenerate ground-state
    multiplet and its momentum quantum numbers.

    Args:
        sample_size: sample size (default ``[3, 4]``).
        params: model parameters (default
            :data:`params_Sun_Gu_Katsura_Sarma`).
        filling_fraction: filling per flattened vertex (default 1/3, i.e.
            ν = 2/3 per band).
        nev: eigenvalues per sector.

    Returns:
        Symmetry_Resolved_ED_Data: the ED data with all sectors scanned.
    """
    if sample_size is None:
        sample_size = [3, 4]
    if params is None:
        params = dict(params_Sun_Gu_Katsura_Sarma)
    model = build_zero_flux_fermionic_fci_second_quantized_model(
        sample_size=sample_size, params=params
    )

    lattice = model.lattice
    n_filled = int(lattice.n_site * Fraction(filling_fraction))
    print(
        f"[ED] Fermionic FCI: {sample_size[0]}×{sample_size[1]} "
        f"checkerboard, {lattice.n_site} sites, {n_filled} fermions "
        f"(filling {filling_fraction})"
    )

    symmetry_group = build_translation_group(lattice)
    ed_data = build_ed_data(
        model,
        filling_fraction=Fraction(filling_fraction),
        symmetry_group=symmetry_group,
    )

    ed_scan(ed_data, nev=nev, mode="matrix")

    print("\n" + "=" * 70)
    print(f"  Fermionic FCI — checkerboard {sample_size[0]}×{sample_size[1]}")
    print(
        f"  Full Hilbert space dim: {math.comb(lattice.n_site, n_filled)}"
    )
    print(
        f"  Orbits: {len(ed_data.orbit_catalog.representative_mask_list)}"
    )
    print(
        f"  Irreps: {len(ed_data.irrep_list)}  (momenta (k₁,k₂))"
    )
    print("=" * 70)

    all_vals: list[float] = []
    all_info: list[tuple[int, Any, int]] = []
    for (irrep_idx, (vals, _)) in ed_data.ed_scan_res.items():
        for (e_idx, v) in enumerate(vals):
            all_vals.append(float(v))
            all_info.append((irrep_idx, ed_data.irrep_list[irrep_idx].label, e_idx))
    perm = sorted(range(len(all_vals)), key=lambda k: all_vals[k])

    print(f"\n--- Lowest {min(15, len(all_vals))} eigenvalues ---")
    for i in range(min(15, len(all_vals))):
        ii, label, ei = all_info[perm[i]]
        print(
            f"  E{i + 1} = {all_vals[perm[i]]:.10f}  (k = {label!r}, #{ei + 1})"
        )

    if len(all_vals) >= 3:
        d12 = all_vals[perm[1]] - all_vals[perm[0]]
        d23 = all_vals[perm[2]] - all_vals[perm[1]]
        d34 = (
            all_vals[perm[3]] - all_vals[perm[2]]
            if len(all_vals) >= 4
            else float("inf")
        )
        print(f"\n  ΔE₁₂ = {d12:.8f}")
        print(f"  ΔE₂₃ = {d23:.8f}")
        print(f"  ΔE₃₄ = {d34:.8f} (many-body gap)")
        if d12 < 0.15 and d23 < 0.15:
            print(
                "  ✓ Three nearly-degenerate GS detected → FCI signature "
                "(GSD=3, finite-size split)!"
            )
            print(
                f"  GS sectors: {ed_data.irrep_list[all_info[perm[0]][0]].label!r}, "
                f"{ed_data.irrep_list[all_info[perm[1]][0]].label!r}, "
                f"{ed_data.irrep_list[all_info[perm[2]][0]].label!r}"
            )

    return ed_data


def test_fermionic_fci_spectrum_flow(
    *,
    sample_size: list[int] | None = None,
    params: dict[str, Any] | None = None,
    filling_fraction: Fraction | int | float = Fraction(1, 3),
    mode: str = "sectors",
    fci_sectors: list[tuple[int, int]] | None = None,
    flux_direction: int | None = None,
    twisted_phases_over_2π_list: list[float] | None = None,
):
    """Spectrum flow for the fermionic FCI on checkerboard.

    Args:
        sample_size: sample size (default ``[3, 4]``).
        params: model parameters (default
            :data:`params_Sun_Gu_Katsura_Sarma`).
        filling_fraction: filling per flattened vertex.
        mode: ``"sectors"`` or ``"identity"``.
        fci_sectors: momentum sector tuples of the FCI GS multiplet.
        flux_direction: **1-based** flux direction (default: first
            direction whose sample size is divisible by the GSD, 3).
        twisted_phases_over_2π_list: flux values to scan.

    Returns:
        SimpleNamespace: the result of :func:`flux_spectrum_flow`.
    """
    if sample_size is None:
        sample_size = [3, 4]
    if params is None:
        params = dict(params_Sun_Gu_Katsura_Sarma)
    if twisted_phases_over_2π_list is None:
        twisted_phases_over_2π_list = list(np.linspace(0.0, 1.0, 9))
    if fci_sectors is None:
        fci_sectors = default_fci_sectors_fermionic(sample_size)
    if flux_direction is None:
        flux_direction = next(
            (d for d in range(1, len(sample_size) + 1) if sample_size[d - 1] % 3 == 0),
            1,
        )

    model = build_zero_flux_fermionic_fci_second_quantized_model(
        sample_size=sample_size, params=params
    )
    lattice = model.lattice
    n_filled = int(lattice.n_site * Fraction(filling_fraction))
    print(
        f"[ED] Fermionic FCI: {sample_size[0]}×{sample_size[1]} "
        f"checkerboard, {lattice.n_site} sites, {n_filled} fermions "
        f"(filling {filling_fraction})"
    )

    if mode not in ("identity", "sectors"):
        raise ValueError("mode must be 'identity' or 'sectors'.")
    is_identity = mode == "identity"
    labels: Any = "identity" if is_identity else fci_sectors
    tag = "identity" if is_identity else "sectors"

    import os

    from ..observables.spectrum_flow import flux_spectrum_flow

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    result = flux_spectrum_flow(
        model,
        labels,
        filling_fraction=Fraction(filling_fraction),
        flux_direction=flux_direction,
        twisted_phases_over_2π_list=twisted_phases_over_2π_list,
        nev=5,
        fig_path=os.path.join(
            project_root,
            "figures",
            f"fermionic_FCI_spectrum_flow_{tag}_{sample_size}.svg",
        ),
        checkpoint_dir=os.path.join(project_root, "checkpoints"),
    )
    return result


def test_fermionic_fci_charge_pump(
    *,
    sample_size: list[int] | None = None,
    params: dict[str, Any] | None = None,
    filling_fraction: Fraction | int | float = Fraction(1, 3),
    mode: str = "sectors",
    fci_sectors: list[tuple[int, int]] | None = None,
    flux_direction: int | None = None,
    polarization_direction: int = 2,
    twisted_phases_over_2π_list: list[float] | None = None,
    atol: float = 0.10,
):
    """Fractional charge pump for the fermionic FCI on checkerboard.

    For ν = 2/3 filling of the lower Chern band, each of the three
    polarization branches should wind by ΔQ ≈ 2/3 over one flux quantum,
    summing to 2 (integer).

    Args:
        sample_size: sample size (default ``[3, 4]``).
        params: model parameters (default
            :data:`params_Sun_Gu_Katsura_Sarma`).
        filling_fraction: filling per flattened vertex.
        mode: ``"sectors"`` or ``"identity"``.
        fci_sectors: the FCI GS multiplet sector labels.
        flux_direction: **1-based** flux direction (default: first
            direction with sample size divisible by the GSD, 3).
        polarization_direction: **1-based** transverse polarization
            direction (default 2).
        twisted_phases_over_2π_list: flux values to scan.
        atol: tolerance for the charge quantization.

    Returns:
        SimpleNamespace: the result of :func:`flux_charge_pump`.
    """
    if sample_size is None:
        sample_size = [3, 4]
    if params is None:
        params = dict(params_Sun_Gu_Katsura_Sarma)
    if twisted_phases_over_2π_list is None:
        twisted_phases_over_2π_list = list(np.linspace(0.0, 1.0, 9))
    if fci_sectors is None:
        fci_sectors = default_fci_sectors_fermionic(sample_size)
    if flux_direction is None:
        flux_direction = next(
            (d for d in range(1, len(sample_size) + 1) if sample_size[d - 1] % 3 == 0),
            1,
        )

    model = build_zero_flux_fermionic_fci_second_quantized_model(
        sample_size=sample_size, params=params
    )
    lattice = model.lattice
    n_filled = int(lattice.n_site * Fraction(filling_fraction))
    print(
        f"[ED] Fermionic FCI: {sample_size[0]}×{sample_size[1]} "
        f"checkerboard, {lattice.n_site} sites, {n_filled} fermions "
        f"(filling {filling_fraction})"
    )

    if mode not in ("identity", "sectors"):
        raise ValueError("mode must be 'identity' or 'sectors'.")
    is_identity = mode == "identity"
    labels: Any = "identity" if is_identity else fci_sectors
    tag = "identity" if is_identity else "sectors"

    import os

    from ..observables.charge_pump import flux_charge_pump

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    result = flux_charge_pump(
        model,
        labels,
        filling_fraction=Fraction(filling_fraction),
        flux_direction=flux_direction,
        polarization_direction=polarization_direction,
        twisted_phases_over_2π_list=twisted_phases_over_2π_list,
        nev_per_sector=1,
        fig_path=os.path.join(
            project_root,
            "figures",
            f"fermionic_FCI_charge_pump_{tag}_{sample_size}.svg",
        ),
        checkpoint_dir=os.path.join(project_root, "checkpoints"),
    )

    if (
        mode == "sectors"
        and abs(twisted_phases_over_2π_list[-1] - 1.0) < 1e-12
    ):
        pumped = np.abs(result.pumped_charges)
        expected_q = 2 / 3
        print(f"[ED] Pumped charges |ΔQ| = {pumped}")
        print(
            "[ED] "
            + (
                f"PASS: all branches pumped ≈ {expected_q} charge."
                if np.all(np.abs(pumped - expected_q) < atol)
                else f"FAIL: a branch deviates from {expected_q} by more than atol."
            )
        )
    return result


def build_fermionic_fci_ed_data(
    *,
    sample_size: list[int] | None = None,
    params: dict[str, Any] | None = None,
    filling_fraction_per_band: Fraction | int | float = Fraction(1, 3),
):
    """Build the symmetry-resolved ED data for the fermionic checkerboard FCI.

    Convenience wrapper used by the benchmark and the examples.

    Args:
        sample_size: sample size (default ``[3, 4]``).
        params: model parameters (default
            :data:`params_Sun_Gu_Katsura_Sarma`).
        filling_fraction_per_band: filling per flatband (default 1/3,
            i.e. the ν = 1/3 FCI); the vertex filling is half of this.

    Returns:
        Symmetry_Resolved_ED_Data: ED data with the translation group.
    """
    if sample_size is None:
        sample_size = [3, 4]
    if params is None:
        params = dict(params_Sun_Gu_Katsura_Sarma)
    model = build_zero_flux_fermionic_fci_second_quantized_model(
        sample_size=sample_size, params=params
    )
    filling_fraction = (
        Fraction(filling_fraction_per_band) / model.lattice.n_sub
    )
    symmetry_group = build_translation_group(model.lattice)
    return build_ed_data(
        model, filling_fraction=filling_fraction, symmetry_group=symmetry_group
    )


__all__ = [
    "params_Sun_Gu_Katsura_Sarma",
    "my_optimal_param",
    "CHECKERBOARD_TPP_DENOMINATOR",
    "build_fermionic_checkerboard_tb_model",
    "build_zero_flux_fermionic_fci_second_quantized_model",
    "build_fermionic_fci_ed_data",
    "build_phase_explore_fermionic_checkerboard_model",
    "default_fci_sectors_fermionic",
    "test_fermionic_fci_full_ed",
    "test_fermionic_fci_spectrum_flow",
    "test_fermionic_fci_charge_pump",
]


def build_phase_explore_fermionic_checkerboard_model(
    params: dict[str, Any],
    sample_size: list[int],
    filling_fraction_per_band: Fraction | int | float,
):
    """Phase-exploration fermionic checkerboard model builder.

    Faithful port of ``_build_fermionic_checkerboard_model`` from the Julia
    ``test/fermionic_fci_phase_explore.jl``: the hoppings are built from
    **pre-flipped** parameters (``t = -1`` etc., i.e. no ``flip_bands``
    step) and all density-density interactions are scaled by
    ``params["λ"]``.

    Args:
        params: model parameters (must contain ``"t"``, ``"t'_1"``,
            ``"t'_2"``, ``"t''"``, ``"phi_over_2pi"``, ``"V1"``, ``"V2"``,
            ``"V3"``, ``"λ"``).
        sample_size: the sample size.
        filling_fraction_per_band: filling per flatband.

    Returns:
        tuple: ``(model, ed_data, n_filled, filling_fraction)`` — the
        second-quantized model and the symmetry-resolved ED data with the
        translation group.
    """
    p = dict(params)
    r_data = initialize_real_space_lattice(
        lattice_name="checkerboard",
        sample_size=sample_size,
        brav_vec_list=[[1.0, 0.0], [0.0, 1.0]],
        sub_crys_list=[[0.5, 0], [0.0, 0.5]],
        pbc_indicator=[True, True],
    )
    tb_model = initialize_real_space_tightbinding_model(
        r_data, model_name="checkerboard"
    )

    t, t1_1, t1_2, t2 = p["t"], p["t'_1"], p["t'_2"], p["t''"]
    phi = 2 * math.pi * p["phi_over_2pi"]

    # NN hoppings (inter-sublattice, complex — staggered flux)
    add_hopping_term(
        tb_model, ((((0, 0), 1), ((0, 0), 2)), -t * cmath.exp(-1j * phi))
    )
    add_hopping_term(
        tb_model, ((((0, 0), 1), ((1, 0), 2)), -t * cmath.exp(1j * phi))
    )
    add_hopping_term(
        tb_model, ((((0, 0), 2), ((0, 1), 1)), -t * cmath.exp(-1j * phi))
    )
    add_hopping_term(
        tb_model, ((((1, 0), 2), ((0, 1), 1)), -t * cmath.exp(1j * phi))
    )

    # NNN hoppings (intra-sublattice, real, anisotropic)
    add_hopping_term(tb_model, ((((0, 0), 1), ((1, 0), 1)), -t1_1))
    add_hopping_term(tb_model, ((((0, 0), 1), ((0, 1), 1)), -t1_2))
    add_hopping_term(tb_model, ((((0, 0), 2), ((1, 0), 2)), -t1_2))
    add_hopping_term(tb_model, ((((0, 0), 2), ((0, 1), 2)), -t1_1))

    # NNNN hoppings (intra-sublattice, diagonal, real)
    add_hopping_term(tb_model, ((((0, 0), 1), ((1, 1), 1)), -t2))
    add_hopping_term(tb_model, ((((0, 0), 2), ((1, 1), 2)), -t2))
    add_hopping_term(tb_model, ((((1, 0), 2), ((0, 1), 2)), -t2))
    add_hopping_term(tb_model, ((((0, 1), 1), ((1, 0), 1)), -t2))

    lattice = tb_model.lattice
    n_site = lattice.n_site
    n_sub = lattice.n_sub
    print(
        f"[ED] Checkerboard TB model: {sample_size} unit cells × {n_sub} "
        f"sublattices = {n_site} sites"
    )

    bilinear_terms = generate_bilinear_terms(
        tb_model, twisted_phases_over_2π=[0.0] * lattice.dim
    )

    V1, V2, V3 = p["V1"], p["V2"], p["V3"]
    V1 *= p["λ"]
    V2 *= p["λ"]
    V3 *= p["λ"]

    L1, L2 = lattice.sample_size
    density_terms: list[tuple[int, int, complex]] = []

    for (i_from, (cell_from, sub_from)) in enumerate(lattice.site_list, start=1):
        for (i_to, (cell_to, sub_to)) in enumerate(lattice.site_list, start=1):
            if i_from == i_to:
                continue
            cell_shift = tuple(
                (c2 - c1) % L
                for c1, c2, L in zip(cell_from, cell_to, lattice.sample_size)
            )

            if (
                V1 != 0.0
                and sub_from == 1
                and sub_to == 2
                and (
                    cell_shift == (0, 0)
                    or cell_shift == (1, 0)
                    or cell_shift == (0, L2 - 1)
                    or cell_shift == (1, L2 - 1)
                )
            ):
                density_terms.append((i_from, i_to, complex(V1)))

            if (
                V2 != 0.0
                and sub_from == sub_to
                and (cell_shift == (1, 0) or cell_shift == (0, 1))
            ):
                density_terms.append((i_from, i_to, complex(V2)))

            if (
                V3 != 0.0
                and sub_from == sub_to
                and (cell_shift == (1, 1) or cell_shift == (1, L2 - 1))
            ):
                density_terms.append((i_from, i_to, complex(V3)))

    model = Real_Space_Second_Quantized_Model(
        params=p,
        lattice=lattice,
        tb_model=tb_model,
        particle_statistics=Particle_Statistics.FERMIONIC,
        bilinear_terms=bilinear_terms,
        density_density_terms=density_terms,
    )

    filling_fraction = Fraction(filling_fraction_per_band) / lattice.n_sub
    print(
        f"[ED] Flatband filling: {filling_fraction_per_band}, flattened "
        f"graph filling: {filling_fraction}"
    )
    n_filled = int(lattice.n_site * filling_fraction)
    print(
        f"[ED] Fermionic FCI: {sample_size[0]}×{sample_size[1]} "
        f"checkerboard, {lattice.n_site} sites, {n_filled} fermions "
        f"(filling {filling_fraction})"
    )

    symmetry_group = build_translation_group(lattice)
    ed_data = build_ed_data(
        model, filling_fraction=filling_fraction, symmetry_group=symmetry_group
    )

    print("\n" + "=" * 70)
    print(f"  Full Hilbert space dim: {math.comb(lattice.n_site, n_filled)}")
    print(
        f"  Orbits: {len(ed_data.orbit_catalog.representative_mask_list)}"
    )
    print(f"  Irreps: {len(ed_data.irrep_list)}  (momenta (k₁,k₂))")
    print("=" * 70)

    print(f"[ED] Scaled interaction strengths: V1={V1}, V2={V2}, V3={V3}")

    return model, ed_data, n_filled, filling_fraction


#: Fine-tuned parameters for the fermionic ν = 1/3 FCI on checkerboard
#: (Julia ``my_optimal_param``; ``λ`` scales all V → λV).
my_optimal_param: dict[str, float] = {
    "t": -1.0,
    "t'_1": -1.4 / (2 + math.sqrt(2)),
    "t'_2": 1.4 / (2 + math.sqrt(2)),
    "t''": -1.0 / (2 + 2 * math.sqrt(2)),
    "phi_over_2pi": 1 / 8,
    "V1": 2.0,
    "V2": 0.45,
    "V3": 0.2,
    "λ": 2.8,
}
