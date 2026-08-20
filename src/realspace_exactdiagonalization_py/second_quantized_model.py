"""Second-quantized model data structures.

Faithful port of ``src/second_quantized_model.jl`` from the Julia package
``RealSpace_ExactDiagonalization.jl``.

The central object is :class:`Real_Space_Second_Quantized_Model`, which pairs
a real-space lattice with the two lists of operator terms that define a
short-range interacting Hamiltonian,

.. math::
   H = \\sum_{(i,j,t) \\in \\mathrm{bilinear}} t\\,a_i^\\dagger a_j
     + \\sum_{(i,j,V) \\in \\mathrm{density}} V\\, n_i n_j,

where :math:`a^{\\dagger}, a` denote canonical creation/annihilation
operators of either hard-core bosons or spinless fermions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - imports used for typing only
    from tightbinding_py.lattice import Real_Space_Lattice
    from tightbinding_py.tb_model import Real_Space_TightBinding_Model


class Particle_Statistics(Enum):
    """Statistics of the particles (or of the model).

    Julia counterpart: the ``MLStyle.@data`` sum type ``Bosonic()`` /
    ``Fermionic()``.  Python members:

    - ``BOSONIC`` — hard-core bosons (no sign factors).
    - ``FERMIONIC`` — spinless fermions (permutation parity and
      Jordan–Wigner string signs are injected in the symmetry actions and
      hopping phases).
    """

    BOSONIC = "Bosonic"
    FERMIONIC = "Fermionic"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Particle_Statistics.{self.name}"


@dataclass
class Real_Space_Second_Quantized_Model:
    """Short-range real-space second-quantized model.

    Julia counterpart: ``mutable struct Real_Space_Second_Quantized_Model{T}
    <: Second_Quantized_Model`` (fields are mutable to support in-place
    twisted-phase updates).

    Attributes:
        params: model parameters, which can include hopping amplitudes,
            interaction strengths, flux values, etc. depending on the model.
            For example, for the Haldane honeycomb model one may use
            ``{"t": 1.0, "t'": 0.6, "t''": -0.58, "phi_over_2pi": 0.2,
            "V1": 0.2, "V2": 0.1}``.
        lattice: the underlying real-space lattice
            (:class:`tightbinding_py.lattice.Real_Space_Lattice`).
        tb_model: the underlying tight-binding model
            (:class:`tightbinding_py.tb_model.Real_Space_TightBinding_Model`).
        particle_statistics: statistics of the particles, i.e.
            :attr:`Particle_Statistics.BOSONIC` or
            :attr:`Particle_Statistics.FERMIONIC`.
        bilinear_terms: bilinear terms of the Hamiltonian; each entry is a
            tuple ``(i, j, t)`` representing :math:`t\\, a_i^\\dagger a_j`,
            where :math:`a` denotes canonical creation/annihilation operators
            of either bosons or fermions (site indices are **1-based**).
        density_density_terms: density-density interaction terms; each entry
            is a tuple ``(i, j, v)`` representing :math:`v\\,n_i n_j`
            (site indices are **1-based**).
    """

    params: dict[str, Any]
    lattice: "Real_Space_Lattice"
    tb_model: "Real_Space_TightBinding_Model"
    particle_statistics: Particle_Statistics
    bilinear_terms: list[tuple[int, int, complex]] = field(default_factory=list)
    density_density_terms: list[tuple[int, int, complex]] = field(
        default_factory=list
    )


def update_second_quantized_model_with_twisted_phases(
    second_quantized_model: Real_Space_Second_Quantized_Model,
    *,
    twisted_phases_over_2π: list[float],
) -> Real_Space_Second_Quantized_Model:
    """**In-place** update of the bilinear hopping terms with twisted phases.

    Julia counterpart: ``update_second_quantized_model_with_twisted_phases!``.

    Delegates to ``TightBinding.generate_bilinear_terms``, which applies the
    Peierls phase :math:`\\exp(i\\,2\\pi\\,\\theta_d\\,w_d)` to every hopping
    crossing a periodic boundary in direction :math:`d` with winding number
    :math:`w_d`.

    The model's ``bilinear_terms`` field is replaced and the lattice's
    ``twisted_phases_over_2π`` is updated.  Density-density terms and all
    other fields are unchanged.

    Args:
        second_quantized_model: the model to update in-place.
        twisted_phases_over_2π: twisted phases :math:`\\theta/(2\\pi)` along
            each periodic direction; must have length ``lattice.dim``.

    Returns:
        Real_Space_Second_Quantized_Model: the same (mutated) model object.
    """
    lattice = second_quantized_model.lattice
    if len(twisted_phases_over_2π) != lattice.dim:
        raise ValueError(
            f"twisted_phases_over_2π must have length {lattice.dim}."
        )

    from tightbinding_py.utils import generate_bilinear_terms

    second_quantized_model.bilinear_terms = generate_bilinear_terms(
        second_quantized_model.tb_model,
        twisted_phases_over_2π=twisted_phases_over_2π,
    )
    second_quantized_model.lattice.twisted_phases_over_2π = list(
        twisted_phases_over_2π
    )

    return second_quantized_model


__all__ = [
    "Particle_Statistics",
    "Real_Space_Second_Quantized_Model",
    "update_second_quantized_model_with_twisted_phases",
]
