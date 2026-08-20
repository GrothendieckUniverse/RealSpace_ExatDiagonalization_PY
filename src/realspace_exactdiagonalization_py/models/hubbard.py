"""Spinful Fermi-Hubbard model on the square lattice.

Faithful port of the model builder in the Julia
``examples/fermion_hubbard_square.jl`` and ``benchmark/benchmark.jl``.

Spinful fermions are handled by flattening the spin degree of freedom into
an interleaved graph: spatial site i↑ → vertex ``2i-1``, i↓ → vertex
``2i``.  With one sublattice per spin and ``sub_crys = [0,0]`` for both,
the site index of (cell, spin) is ``2*i_cell - 2 + spin``.
"""

from __future__ import annotations

from fractions import Fraction

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
from ..symmetry_resolved_ed import build_ed_data, build_translation_group


def build_spinful_hubbard_second_quantized_model(
    sample_size: list[int], t: float = 1.0, U: float = 8.0
) -> Real_Space_Second_Quantized_Model:
    """Build the spin-½ Fermi-Hubbard model on a square lattice.

    Args:
        sample_size: ``[Lx, Ly]`` spatial sample size.
        t: hopping amplitude.
        U: on-site Hubbard repulsion.

    Returns:
        Real_Space_Second_Quantized_Model: the second-quantized model
        (fermionic) with spin-flattened vertices.
    """
    Lx, Ly = sample_size

    # spatial lattice (one sublattice, spinless)
    r_data = initialize_real_space_lattice(
        sample_size=sample_size,
        brav_vec_list=[[1.0, 0.0], [0.0, 1.0]],
        sub_crys_list=[[0.0, 0.0]],
        lattice_name="Square_Hubbard",
        pbc_indicator=[True, True],
    )
    tb_spinless = initialize_real_space_tightbinding_model(
        r_data, model_name="hubbard_spinless"
    )
    add_hopping_term(tb_spinless, ((((0, 0), 1), ((1, 0), 1)), complex(-t)))
    add_hopping_term(tb_spinless, ((((0, 0), 1), ((0, 1), 1)), complex(-t)))

    n_spatial = r_data.n_site  # Lx * Ly

    # spinful lattice: 2 "sublattices" = ↑, ↓
    lattice = initialize_real_space_lattice(
        sample_size=sample_size,
        brav_vec_list=[[1.0, 0.0], [0.0, 1.0]],
        sub_crys_list=[[0.0, 0.0], [0.0, 0.0]],
        lattice_name="Square_Hubbard_Spinful",
        pbc_indicator=[True, True],
    )
    n_site = lattice.n_site  # 2 * Lx * Ly

    def _sv(x: int, y: int, s: int) -> int:
        """Linear vertex index of (cell (x, y), spin s ∈ {1↑, 2↓})."""
        return lattice.site_to_index_map[((x, y), s)]

    bilinear_terms: list[tuple[int, int, complex]] = []
    for ((site_from, site_to), tamp) in tb_spinless.full_hopping_map.items():
        x1, y1 = site_from[0]
        x2, y2 = site_to[0]
        bilinear_terms.append((_sv(x1, y1, 1), _sv(x2, y2, 1), complex(tamp)))
        bilinear_terms.append((_sv(x1, y1, 2), _sv(x2, y2, 2), complex(tamp)))

    density_terms: list[tuple[int, int, complex]] = [
        (_sv(x, y, 1), _sv(x, y, 2), complex(U))
        for y in range(Ly)
        for x in range(Lx)
    ]

    tb_model = initialize_real_space_tightbinding_model(
        lattice, model_name="Hubbard"
    )
    return Real_Space_Second_Quantized_Model(
        params={"t": t, "U": U},
        lattice=lattice,
        tb_model=tb_model,
        particle_statistics=Particle_Statistics.FERMIONIC,
        bilinear_terms=bilinear_terms,
        density_density_terms=density_terms,
    )


def build_spinful_hubbard_ed_data(
    sample_size: list[int], t: float = 1.0, U: float = 8.0
):
    """Build the symmetry-resolved ED data of the spinful Hubbard model.

    Half-filling: :math:`N_e = L_x L_y` particles on the
    :math:`2 L_x L_y` flattened vertices (filling fraction 1/2).

    Args:
        sample_size: ``[Lx, Ly]`` spatial sample size.
        t: hopping amplitude.
        U: on-site Hubbard repulsion.

    Returns:
        Symmetry_Resolved_ED_Data: ED data with the translation group
        (which moves both spin flavors together).
    """
    model = build_spinful_hubbard_second_quantized_model(sample_size, t, U)
    n_site = model.lattice.n_site
    n_filled = sample_size[0] * sample_size[1]  # half filling
    symmetry_group = build_translation_group(model.lattice)
    return build_ed_data(
        model,
        filling_fraction=Fraction(n_filled, n_site),
        symmetry_group=symmetry_group,
    )


__all__ = [
    "build_spinful_hubbard_second_quantized_model",
    "build_spinful_hubbard_ed_data",
]
