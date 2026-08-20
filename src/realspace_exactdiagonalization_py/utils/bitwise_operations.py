"""Bitwise operations on occupation-number bitmasks.

Faithful port of ``BitWise_Operations`` from the Julia package
``RealSpace_ExactDiagonalization.jl`` (``src/bitwise_operations.jl``).

A Fock-space configuration :math:`|n_1,\\dots,n_N\\rangle` with hard-core
occupations :math:`n_i\\in\\{0,1\\}` is encoded as the unsigned integer

.. math::
   m = \\sum_{i=1}^{N} n_i\\,2^{\\,i-1},

where bit ``i - 1`` (0-based) corresponds to graph vertex ``i`` (1-based).
All masks used by this package fit in 64 bits (:math:`N \\le 63`), so
``numpy.uint64`` kernels may be used inside Numba-compiled hot loops while the
public API uses ordinary (arbitrary-precision) Python integers.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
#: Bit representation of a Fock state in an integer, used for state indexing.
Mask = int

#: ``Complex(1.0)`` — the multiplicative identity phase.
COMPLEX_ONE = 1.0 + 0.0j

#: Maximum number of graph vertices supported by the 64-bit kernels.
MAX_N_SITE_UINT64 = 63


def bitmask_of_site(i: int) -> Mask:
    """Bit representation for the occupied/empty two-state vertex.

    Args:
        i: the **1-based** linear index of the vertex in the graph.

    Returns:
        Mask: the integer whose ``i - 1``-th bit (0-based) is set.
    """
    return Mask(1) << (i - 1)


def occupy_site_for_mask(m: Mask, i: int) -> Mask:
    """Occupy vertex ``i`` for mask ``m``.

    Args:
        m: input occupation mask.
        i: **1-based** linear index of the vertex to occupy.

    Returns:
        Mask: ``m`` with the ``i - 1``-th bit set.
    """
    return m | bitmask_of_site(i)


def empty_site_for_mask(m: Mask, i: int) -> Mask:
    """Empty vertex ``i`` for mask ``m``.

    Args:
        m: input occupation mask.
        i: **1-based** linear index of the vertex to empty.

    Returns:
        Mask: ``m`` with the ``i - 1``-th bit cleared.
    """
    return m & ~bitmask_of_site(i)


def is_site_occupied(m: Mask, i: int) -> bool:
    """Whether vertex ``i`` is occupied in mask ``m``.

    Args:
        m: input occupation mask.
        i: **1-based** linear index of the vertex.

    Returns:
        bool: ``True`` if the ``i - 1``-th bit of ``m`` is set.
    """
    return (m & bitmask_of_site(i)) != 0


def is_site_empty(m: Mask, i: int) -> bool:
    """Whether vertex ``i`` is empty in mask ``m``.

    Args:
        m: input occupation mask.
        i: **1-based** linear index of the vertex.

    Returns:
        bool: ``True`` if the ``i - 1``-th bit of ``m`` is clear.
    """
    return (m & bitmask_of_site(i)) == 0


def n_occupied_for_mask(m: Mask) -> int:
    """Number of occupied vertices in mask ``m`` (population count).

    Args:
        m: input occupation mask.

    Returns:
        int: the number of set bits in ``m``.
    """
    return m.bit_count()


def filled_site_iter_for_mask(m: Mask, n_site: int) -> Iterator[int]:
    """Iterator of occupied site indices from a bit mask ``m``.

    Unlike ``decode_bit_mask_to_configuration``, this generator avoids
    explicit construction of the configuration vector to save allocations.

    Args:
        m: input occupation mask; the ``i``-th bit (1-based) indicates whether
            vertex ``i`` is occupied.
        n_site: total number of sites in the lattice, which determines how
            many bits need checking in the mask.

    Returns:
        Iterator[int]: yields the **1-based** indices of occupied sites.
        For example ``m = 0b10110`` and ``n_site = 5`` yields ``2, 3, 5``.
    """
    return (i for i in range(1, n_site + 1) if is_site_occupied(m, i))


def empty_site_iter_for_mask(m: Mask, n_site: int) -> Iterator[int]:
    """Iterator of empty site indices from a bit mask ``m``.

    Unlike ``decode_bit_mask_to_configuration``, this generator avoids
    explicit construction of the configuration vector to save allocations.

    Args:
        m: input occupation mask; the ``i``-th bit (1-based) indicates whether
            vertex ``i`` is occupied.
        n_site: total number of sites in the lattice, which determines how
            many bits need checking in the mask.

    Returns:
        Iterator[int]: yields the **1-based** indices of empty sites.
        For example ``m = 0b10110`` and ``n_site = 5`` yields ``1, 4``.
    """
    return (i for i in range(1, n_site + 1) if is_site_empty(m, i))


def encode_configuration_to_bit_mask(occ: Iterable[int]) -> Mask:
    """Bit representation of a configuration ``occ``.

    Args:
        occ: vector of occupied site linear indices; each element is the
            **1-based** linear index of an occupied site in the lattice.

    Returns:
        Mask: the occupation bitmask with one set bit per entry of ``occ``.
    """
    m = Mask(0)
    for i in occ:
        m = occupy_site_for_mask(m, int(i))
    return m


def decode_bit_mask_to_configuration(m: Mask, n_site: int) -> list[int]:
    """Configuration (vector of occupied site linear indices) from mask ``m``.

    Inverse operation of :func:`encode_configuration_to_bit_mask`.

    Args:
        m: input occupation mask.
        n_site: total number of sites in the lattice (bits to scan).

    Returns:
        list[int]: **1-based** occupied site indices in ascending order.
    """
    occ: list[int] = []
    for i in range(1, n_site + 1):
        if is_site_occupied(m, i):
            occ.append(i)
    return occ


def decode_bit_mask_to_configuration_inplace(
    occ: list[int], m: Mask, n_site: int
) -> list[int]:
    """**In-place** update of ``occ`` from bit mask ``m``.

    The existing entries of ``occ`` are overwritten (no allocation); the
    caller must guarantee ``len(occ) == n_occupied_for_mask(m)`` (this is
    asserted). Inverse operation of :func:`encode_configuration_to_bit_mask`.

    Args:
        occ: pre-allocated list of occupied site indices to be overwritten;
            its length must equal the population count of ``m``.
        m: input occupation mask.
        n_site: total number of sites in the lattice (bits to scan).

    Returns:
        list[int]: the same list ``occ``, now holding the **1-based**
        occupied site indices of ``m`` in ascending order.
    """
    assert len(occ) == n_occupied_for_mask(m), (
        "decode_bit_mask_to_configuration_inplace: length(occ) must equal "
        "the number of occupied sites in the mask."
    )
    n = 0
    for i in range(1, n_site + 1):
        if is_site_occupied(m, i):
            occ[n] = i  # overwrite the existing entries of `occ`
            n += 1
    return occ


def as_uint64_array(mask_list: list[Mask]) -> np.ndarray:
    """Convert a list of Python-int masks into a ``uint64`` numpy array.

    Helper used at the boundary between the pure-Python API and the
    Numba-compiled kernels.

    Args:
        mask_list: list of occupation masks.

    Returns:
        np.ndarray: 1-D ``uint64`` array of the same masks.
    """
    return np.asarray(mask_list, dtype=np.uint64)


__all__ = [
    "Mask",
    "COMPLEX_ONE",
    "MAX_N_SITE_UINT64",
    "bitmask_of_site",
    "occupy_site_for_mask",
    "empty_site_for_mask",
    "is_site_occupied",
    "is_site_empty",
    "n_occupied_for_mask",
    "filled_site_iter_for_mask",
    "empty_site_iter_for_mask",
    "encode_configuration_to_bit_mask",
    "decode_bit_mask_to_configuration",
    "decode_bit_mask_to_configuration_inplace",
    "as_uint64_array",
]
