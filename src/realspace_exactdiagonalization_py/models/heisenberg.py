"""Spin-½ Heisenberg chain via the hard-core boson (Matsubara–Matsuda) mapping.

Faithful port of the model builder in the Julia
``examples/spin_heisenberg_chain.jl`` and ``benchmark/benchmark.jl``.

The spin Hamiltonian :math:`H = J\\sum_{\\langle i,j\\rangle} \\mathbf S_i\\cdot
\\mathbf S_j` is mapped to hard-core bosons via
:math:`S^z_i = n_i - \\tfrac12`, :math:`S^+_i = b_i^\\dagger`,
:math:`S^-_i = b_i`:

.. math::
   \\mathbf S_i\\cdot\\mathbf S_j = n_i n_j + \\tfrac12(b_i^\\dagger b_j +
   \\mathrm{h.c.}) - \\tfrac12 n_i - \\tfrac12 n_j + \\tfrac14.

Summing over bonds, :math:`\\sum(-\\tfrac12 n_i - \\tfrac12 n_j) = -\\sum_i n_i
= -N_e` and :math:`\\sum \\tfrac14 = N/4`; at half filling :math:`N_e = N/2`
the net constant is :math:`-JN/4`, absorbed via on-site terms
:math:`(i,i,-J/2)` so that the coded Hamiltonian equals the spin
Hamiltonian with no external constant-shift bookkeeping.
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


def build_heisenberg_second_quantized_model(
    N: int, J: float = 1.0
) -> Real_Space_Second_Quantized_Model:
    """Build the spin-½ Heisenberg chain as a hard-core boson model.

    Args:
        N: chain length (PBC along x).
        J: exchange coupling (``J > 0`` = antiferromagnetic).

    Returns:
        Real_Space_Second_Quantized_Model: the second-quantized model
        (bosonic) with bilinear hopping ``J/2`` on each bond and
        density-density terms ``J`` on bonds + ``-J/2`` on sites.
    """
    r_data = initialize_real_space_lattice(
        sample_size=[N, 1],
        brav_vec_list=[[1.0, 0.0], [0.0, 1.0]],
        sub_crys_list=[[0.0, 0.0]],
        lattice_name="1D_Chain",
        pbc_indicator=[True, False],  # PBC only along x
    )
    tb_model = initialize_real_space_tightbinding_model(
        r_data, model_name="Heisenberg_Chain"
    )

    # NN hopping (b†_i b_j + h.c.) with amplitude J/2 along x only.
    # add_hopping_term expands via translation symmetry — one bond suffices.
    add_hopping_term(tb_model, ((((0, 0), 1), ((1, 0), 1)), J / 2))

    lattice = tb_model.lattice
    n_site = lattice.n_site

    bilinear_terms = generate_bilinear_terms(
        tb_model, twisted_phases_over_2π=[0.0] * lattice.dim
    )

    density_terms: list[tuple[int, int, complex]] = []
    # Bond terms: J * n_i * n_j
    for x in range(N):
        i = lattice.site_to_index_map[((x, 0), 1)]
        j = lattice.site_to_index_map[(((x + 1) % N, 0), 1)]
        density_terms.append((i, j, complex(J)))
    # On-site terms: -J/2 * n_i (absorbs -½n_i-½n_j and the +¼ constant)
    for x in range(N):
        i = lattice.site_to_index_map[((x, 0), 1)]
        density_terms.append((i, i, complex(-J / 2)))

    return Real_Space_Second_Quantized_Model(
        params={"J": J, "N_site": N},
        lattice=lattice,
        tb_model=tb_model,
        particle_statistics=Particle_Statistics.BOSONIC,
        bilinear_terms=bilinear_terms,
        density_density_terms=density_terms,
    )


def build_heisenberg_ed_data(N: int, J: float = 1.0):
    """Build the symmetry-resolved ED data of the Heisenberg chain.

    Half-filling sector: :math:`N_e = N/2` (total :math:`S^z = 0`);
    translation symmetry: :math:`\\mathbb{Z}_N`.

    Args:
        N: chain length (must be even for half filling).
        J: exchange coupling.

    Returns:
        Symmetry_Resolved_ED_Data: ED data with the translation group.
    """
    model = build_heisenberg_second_quantized_model(N, J)
    n_site = model.lattice.n_site
    n_filled = N // 2
    symmetry_group = build_translation_group(model.lattice)
    return build_ed_data(
        model,
        filling_fraction=Fraction(n_filled, n_site),
        symmetry_group=symmetry_group,
    )


__all__ = [
    "build_heisenberg_second_quantized_model",
    "build_heisenberg_ed_data",
]
