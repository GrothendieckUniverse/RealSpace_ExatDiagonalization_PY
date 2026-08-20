"""Symmetry-resolved exact diagonalization engine.

Faithful port of ``src/symmetry_resolved_ed.jl`` from the Julia package
``RealSpace_ExactDiagonalization.jl`` — an XDiag-inspired, high-performance
exact-diagonalization engine for hard-core bosons and fermions on arbitrary
real-space lattices.

Two computational modes are supported (following XDiag):

- **``"matrix"`` mode** — precompute the sparse Hamiltonian block of a
  symmetry sector, then diagonalize it (fast, memory-heavy);
- **``"matrixfree"`` mode** — apply :math:`H|\\psi\\rangle` on the fly with
  multithreaded Lanczos.  The projected hopping amplitudes are precomputed
  once into a flat projection table (:math:`\\sim\\mathrm{nnz}` triplets —
  *less* memory than the Julia CanonicalMap Dict) and each matvec is a
  pure gather–scatter kernel (18× faster than re-projecting through
  Numba's typed.Dict, whose probes are ~5× slower than Julia's Dict).

Architecture (identical to the Julia reference):

1. symmetry group :math:`G` → :class:`Symmetry_Operation` (permutation +
   U(1) phases);
2. Gosper's hack → enumerate all bitmasks at fixed particle number;
3. orbit–stabilizer decomposition → :class:`Symmetry_Orbit_Catalog`;
4. 1-D irreps → filter orbits → :class:`Symmetry_Sector_Basis`;
5a. ``"matrix"`` mode → sparse CSR block → ARPACK (:func:`scipy.sparse.linalg.eigsh`);
5b. ``"matrixfree"`` mode → XDiag-style representative tables for O(1)
   lookup + a precomputed projection table and Numba-parallel
   :math:`H|\\psi\\rangle` → ARPACK on a
   :class:`~scipy.sparse.linalg.LinearOperator`.

Key references:

- XDiag: https://github.com/awietek/xdiag / arXiv:2505.02901
- D. N. Sheng et al., *Phys. Rev. Lett.* **107**, 146803 (2011).

The hot loops (bitmask arithmetic, orbit enumeration, matrix construction,
matrix-free :math:`H|\\psi\\rangle`) are compiled with Numba (``njit`` +
``prange``), replacing Julia's ``@inbounds @fastmath`` / ``Threads.@threads``
loops.  All public API names mirror the Julia originals (with the ``!``
mutation suffix dropped, as Python has no such convention).

.. note::
    Site indices are **1-based** everywhere (as in the Julia package);
    ``irrep_idx``/``ed_scan_res`` dictionary keys are **0-based** Python
    indices.
"""

from __future__ import annotations

import cmath
import math
import os
import pickle
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .utils.bitwise_operations import (
    COMPLEX_ONE,
    MAX_N_SITE_UINT64,
    Mask,
    as_uint64_array,
    bitmask_of_site,
    empty_site_for_mask,
    is_site_empty,
    is_site_occupied,
    n_occupied_for_mask,
    occupy_site_for_mask,
)
from .second_quantized_model import (
    Particle_Statistics,
    Real_Space_Second_Quantized_Model,
)

# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# Pure-numpy kernel layer — mirrors the Julia hot loops bit-for-bit
#
# The original Numba-jit layer was removed (A/B benchmark, Heis N=24):
#   catalog    numba 1.25 s  → numpy 1.21 s  (0.97×)
#   H-block    numba 0.23 s  → numpy 0.13 s  (0.56×)
#   proj-table numba 0.23 s  → numpy 0.12 s  (0.53×)
#   matvec     numba 4.15 ms → numpy 2.61 ms (CSR, 0.63×)
# Everything is vectorized over chunks of the fixed-filling basis:
# group actions are byte-lookup gathers, the rep test is a two-pass
# min-reduce, and the matvec is a single sparse-matrix product.
# ═════════════════════════════════════════════════════════════════════════════

def _popcount_u64(x: int) -> int:
    """Population count of a mask (Python int)."""
    return int(x).bit_count()


def _trailing_zeros(x: int) -> int:
    """Number of trailing zero bits of a non-zero mask."""
    return (x & -x).bit_length() - 1


# ---------------------------------------------------------------------------
# Scalar kernels (public scalar API; the vectorized builders below supersede
# them on the hot paths but these stay for single-mask use, e.g. in
# `get_canonical` / `apply_operation_to_mask`).
# ---------------------------------------------------------------------------

def _lin_index_kernel(
    m: np.uint64,
    lin_left: np.ndarray,
    lin_right: np.ndarray,
    n_right: np.int64,
) -> np.int64:
    """O(1) combinadic rank from the split tables (Lin 1990)."""
    return lin_left[int(m) >> int(n_right)] + lin_right[
        int(m) & ((1 << int(n_right)) - 1)
    ]


def _rank_kernel(m: np.uint64, binom_table: np.ndarray) -> np.int64:
    """Combinadic (colexicographic) rank of a fixed-popcount bitmask.

    Port of XDiag's ``rank_combination``:
    ``index(bits) = Σ_j binom(b_j, j+1)`` over the set-bit positions
    ``b_0 < b_1 < …`` (0-based) — a deterministic bijection between
    k-subsets and ``[0, C(n,k))`` needing no hash table.  ``binom_table``
    is a precomputed ``(n_site+1) × (n_filled+1)`` array of binomials.
    """
    idx = 0
    j = 1
    tmp = int(m)
    while tmp != 0:
        lsb = tmp & -tmp
        b = lsb.bit_length() - 1  # 0-based bit position
        idx += int(binom_table[b, j])
        j += 1
        tmp ^= lsb
    return np.int64(idx)


def _binom_table(n_site: int, n_filled: int) -> np.ndarray:
    """Precomputed binomial table ``C(b, j)`` for ``b in 0..n_site``,
    ``j in 0..n_filled`` (0 outside the triangle).

    Args:
        n_site: number of vertices.
        n_filled: particle number.

    Returns:
        np.ndarray: ``(n_site+1, n_filled+1)`` int64 array.
    """
    table = np.zeros((n_site + 1, n_filled + 1), dtype=np.int64)
    for b in range(n_site + 1):
        for j in range(min(b, n_filled) + 1):
            table[b, j] = math.comb(b, j)
    return table


#: LinTable gate: split-table O(1) rank is used for n_site <= this value
#: (memory 2·2^{n/2} Int64 entries; XDiag's ladder caps it at 42).
LIN_TABLE_MAX_N_SITE = 42


def _lin_table(n_site: int, n_filled: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the split-table O(1) combinadic rank (Lin 1990; XDiag LinTable).

    For a mask split into an upper half ``hi`` (``n_left`` bits) and a lower
    half ``lo`` (``n_right`` bits),

    .. math::
        \mathrm{rank}(m) = L[hi] + R[lo],

    with :math:`L[u] = \sum_{u' < u} \binom{n_{\mathrm{right}}}{k - \mathrm{popcount}(u')}`
    (the number of k-subsets whose upper half is strictly below ``u``) and
    :math:`R[v] = \sum_j \binom{b_j}{j+1}` over the set bits of ``v`` (the
    colex rank of ``v`` within its own popcount class).

    Args:
        n_site: number of vertices.
        n_filled: particle number.

    Returns:
        tuple: ``(left_indices, right_indices, n_right)``.
    """
    n_right = n_site // 2
    n_left = n_site - n_right
    size_right = 1 << n_right
    size_left = 1 << n_left
    left_indices = np.zeros(size_left, dtype=np.int64)
    right_indices = np.zeros(size_right, dtype=np.int64)
    for v in range(size_right):
        r = 0
        j = 1
        tmp = v
        while tmp:
            lsb = tmp & -tmp
            b = lsb.bit_length() - 1
            r += math.comb(b, j)
            j += 1
            tmp ^= lsb
        right_indices[v] = r
    acc = 0
    for u in range(size_left):
        left_indices[u] = acc
        pc = u.bit_count()
        rem = n_filled - pc
        if 0 <= rem <= n_right:
            acc += math.comb(n_right, rem)
    return left_indices, right_indices, n_right


def _get_canonical_table_kernel(
    rep_rank_table: np.ndarray,
    rep_sym_table: np.ndarray,
    rep_amp_table: np.ndarray,
    catalog_repr_masks: np.ndarray,
    binom_table: np.ndarray,
    lin_left: np.ndarray,
    lin_right: np.ndarray,
    lin_n_right: np.int64,
    m: np.uint64,
) -> tuple[np.uint64, np.int64, np.complex128]:
    """O(1) canonical lookup from the precomputed representative tables.

    XDiag-style: combinadic rank + three array reads — no hashing, no
    O(|G|) fallback (every fixed-filling mask is covered by construction).
    """
    if int(lin_n_right) >= 0:
        idx = _lin_index_kernel(m, lin_left, lin_right, lin_n_right)
    else:
        idx = _rank_kernel(m, binom_table)
    rep_idx = int(rep_rank_table[idx])
    return catalog_repr_masks[rep_idx], rep_sym_table[idx], np.complex128(
        rep_amp_table[idx]
    )


def _gosper_next_kernel(x: np.uint64) -> np.uint64:
    """Next fixed-popcount mask after ``x`` (Gosper's hack)."""
    x = int(x)
    c = x & -x
    r = x + c
    return np.uint64((((r ^ x) >> 2) // c) | r)


def _first_combination_mask_kernel(n_filled: np.int64) -> np.uint64:
    """Mask with the ``n_filled`` lowest bits set."""
    if int(n_filled) == 0:
        return np.uint64(0)
    return np.uint64((1 << int(n_filled)) - 1)


def _hopping_phase_kernel(
    stats_flag: bool,
    m: np.uint64,
    i_from: np.int64,
    i_to: np.int64,
) -> np.complex128:
    """Fermionic Jordan–Wigner string phase for the hop ``i_from → i_to``.

    Mirrors ``hopping_phase_for_stats``: for fermions the phase is
    :math:`(-1)^{\\#\\{\\text{occupied sites strictly between }i_{\\mathrm{from}}
    \\text{ and }i_{\\mathrm{to}}\\}}`; bosons always get :math:`+1`.
    """
    if not stats_flag or int(i_from) == int(i_to):
        return 1.0 + 0.0j
    lo = min(int(i_from), int(i_to))
    hi = max(int(i_from), int(i_to))
    between = 0
    if hi - lo > 1:
        between = ((1 << (hi - lo - 1)) - 1) << lo
    if int(m).bit_count() if False else _popcount_u64(int(m) & between) % 2 == 1:
        return -1.0 + 0.0j
    return 1.0 + 0.0j


def _is_bit_set(m: np.uint64, i: np.int64) -> bool:
    return (int(m) & (1 << (int(i) - 1))) != 0


def _apply_operation_to_mask_kernel(
    m: np.uint64,
    perm_bits: np.ndarray,
    perms: np.ndarray,
    perm_phases: np.ndarray,
    g_idx: np.int64,
    stats_flag: bool,
) -> tuple[np.uint64, np.complex128]:
    """Apply group element ``g_idx`` to mask ``m`` (single-operation kernel).

    Mirrors ``apply_operation_to_mask`` in Julia exactly: bosons accumulate
    only the per-site U(1) phases; fermions additionally accumulate the
    permutation parity via ``popcount(new_mask >> π(i))``.
    """
    m = int(m)
    g_idx = int(g_idx)
    tmp = m
    new_mask = 0
    phase = 1.0 + 0.0j
    parity = 1
    while tmp != 0:
        lsb = tmp & -tmp
        idx = lsb.bit_length() - 1  # 0-based site index
        p = int(perms[g_idx, idx])
        if stats_flag:
            if _popcount_u64(new_mask >> p) % 2 == 1:
                parity = -parity
        new_mask |= int(perm_bits[g_idx, idx])
        phase *= complex(perm_phases[g_idx, idx])
        tmp ^= lsb
    if stats_flag:
        return np.uint64(new_mask), phase * parity
    return np.uint64(new_mask), phase


def _canonical_representative_kernel(
    m: np.uint64,
    perm_bits: np.ndarray,
    perms: np.ndarray,
    perm_phases: np.ndarray,
    stats_flag: bool,
    identity_idx: np.int64,
) -> tuple[np.uint64, np.int64, np.complex128]:
    """Minimal orbit representative of ``m`` under the whole group (kernel)."""
    repr_mask = int(m)
    best_g = int(identity_idx)
    best_amp = 1.0 + 0.0j
    n_g = perm_bits.shape[0]
    for g_idx in range(n_g):
        shifted, alpha = _apply_operation_to_mask_kernel(
            m, perm_bits, perms, perm_phases, g_idx, stats_flag
        )
        if int(shifted) < repr_mask:
            repr_mask = int(shifted)
            best_g = g_idx
            best_amp = alpha
    return np.uint64(repr_mask), np.int64(best_g), best_amp


# ---------------------------------------------------------------------------
# Vectorized helpers for the chunked builders
# ---------------------------------------------------------------------------

def _make_byte_tables(n_site: int, perm_bits: np.ndarray, perm_phases: np.ndarray):
    """Byte-lookup tables for the whole group action.

    For each group element ``g`` and each input byte position ``b``
    (bits ``8b..8b+7``):
      ``perm_tbl[g, b, v]`` = the uint64 image of the byte ``v``'s bits under ``g``
      ``angle_tbl[g, b, v]`` = Σ_{set bits i of v within byte b} arg(η_g(i))
    Applying ``g`` to a chunk of masks is then ``n_bytes`` ``np.take``
    gathers + OR (and gathers + add for the angle), all at C speed.
    """
    n_bytes = (n_site + 7) // 8
    n_g = perm_bits.shape[0]
    perm_tbl = np.zeros((n_g, n_bytes, 256), dtype=np.uint64)
    angle_tbl = np.zeros((n_g, n_bytes, 256), dtype=np.float64)
    for g in range(n_g):
        for b in range(n_bytes):
            for v in range(256):
                out = 0
                ang = 0.0
                for i in range(8):
                    si = 8 * b + i
                    if si >= n_site:
                        break
                    if (v >> i) & 1:
                        out |= int(perm_bits[g, si])
                        ang += np.angle(perm_phases[g, si])
                perm_tbl[g, b, v] = out
                angle_tbl[g, b, v] = ang
    return perm_tbl, angle_tbl, n_bytes


def _np_unrank_chunk(ranks: np.ndarray, n_site: int, n_filled: int,
                     binom_table: np.ndarray) -> np.ndarray:
    """Colex unranking: masks for a chunk of ranks (vectorized).

    ``rank(m) = Σ_j C(b_j, j+1)`` over set-bit positions ``b_0<b_1<...`` —
    so unranking walks ``j = k..1`` choosing the largest ``b`` with
    ``C(b, j) ≤ r``.
    """
    k = n_filled
    masks = np.zeros(len(ranks), dtype=np.uint64)
    r = ranks.astype(np.int64).copy()
    for j in range(k, 0, -1):
        col = binom_table[:, j]
        b = np.searchsorted(col, r, side="right") - 1
        b = np.minimum(b, n_site - 1)
        masks |= (np.uint64(1) << b.astype(np.uint64))
        r -= binom_table[b, j]
    return masks


def _np_rank(masks: np.ndarray, lin_left: np.ndarray, lin_right: np.ndarray,
             lin_n_right: int) -> np.ndarray:
    """Vectorized O(1) LinTable rank (caller handles the fallback)."""
    lo_mask = (np.uint64(1) << lin_n_right) - 1
    return lin_left[masks >> lin_n_right] + lin_right[masks & lo_mask]


def _np_rank_fallback(masks: np.ndarray, binom_table: np.ndarray,
                      n_site: int) -> np.ndarray:
    """Vectorized combinadic rank for ``n_site > LIN_TABLE_MAX_N_SITE``."""
    n = masks.shape[0]
    rank = np.zeros(n, dtype=np.int64)
    j = np.ones(n, dtype=np.int64)
    tmp = masks.copy()
    for i in range(n_site):
        bit = (tmp & (np.uint64(1) << i)) != 0
        occ = np.nonzero(bit)[0]
        if occ.size:
            rank[occ] += binom_table[i, j[occ]]
            j[occ] += 1
            tmp[occ] ^= (np.uint64(1) << i)
    return rank


def _np_rank_any(masks: np.ndarray, lin_left: np.ndarray,
                 lin_right: np.ndarray, lin_n_right: int,
                 binom_table: np.ndarray, n_site: int) -> np.ndarray:
    if lin_n_right >= 0:
        return _np_rank(masks, lin_left, lin_right, lin_n_right)
    return _np_rank_fallback(masks, binom_table, n_site)


# ---------------------------------------------------------------------------
# Orbit catalog kernel
# ---------------------------------------------------------------------------

def _build_orbit_catalog_kernel(
    n_site: np.int64,
    n_filled: np.int64,
    perm_bits: np.ndarray,
    perms: np.ndarray,
    perm_phases: np.ndarray,
    stats_flag: bool,
    binom_table: np.ndarray,
    lin_left: np.ndarray,
    lin_right: np.ndarray,
    lin_n_right: np.int64,
    rep_rank_table: np.ndarray,
    rep_sym_table: np.ndarray,
    rep_amp_table: np.ndarray,
    chunk: int = 1 << 16,
) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    """Kernel for the orbit–stabilizer decomposition (pure numpy).

    XDiag-style: representatives are found by the ``isrepresentative``
    early-exit test (no ``seen`` set), and each representative's orbit is
    expanded in the SAME pass into the dense representative tables (indexed
    by the combinadic rank).  Every full-basis state is written exactly
    once (orbits are disjoint).

    Vectorized two-pass sweep over the basis in colex order:
      pass 1 — the image of every mask under every ``g`` is computed by
               byte-lookup gathers and min-reduced for the rep test;
      pass 2 — for the found reps only (≈ B/|G| of the chunk), images +
               phases are recomputed and scattered into the tables.
    """
    n_site = int(n_site)
    n_filled = int(n_filled)
    lin_n_right = int(lin_n_right)
    n_g = perm_bits.shape[0]
    n_total = math.comb(n_site, n_filled)

    angle_table = np.angle(perm_phases)
    inv_table = np.zeros((n_g, n_site, n_site), dtype=np.int64)
    if stats_flag:
        for g in range(n_g):
            for i in range(n_site):
                for j in range(i + 1, n_site):
                    if perms[g, i] > perms[g, j]:
                        inv_table[g, i, j] = 1

    repr_list = []
    stab_order_list = []
    stab_gidx_list = []
    stab_phase_list = []

    if n_filled == 0 or n_filled == n_site:
        m = np.uint64((1 << n_filled) - 1)
        gidx_stab, phase_stab = [], []
        for g_idx in range(n_g):
            shifted, alpha = _apply_operation_to_mask_kernel(
                m, perm_bits, perms, perm_phases, g_idx, stats_flag
            )
            if shifted != m:
                continue
            gidx_stab.append(g_idx)
            phase_stab.append(alpha)
        repr_list.append(int(m))
        stab_order_list.append(len(gidx_stab))
        stab_gidx_list.append(np.asarray(gidx_stab, dtype=np.int64))
        stab_phase_list.append(np.asarray(phase_stab, dtype=np.complex128))
        for g_idx in range(n_g):
            shifted, alpha = _apply_operation_to_mask_kernel(
                m, perm_bits, perms, perm_phases, g_idx, stats_flag
            )
            if lin_n_right >= 0:
                idx = _lin_index_kernel(shifted, lin_left, lin_right, lin_n_right)
            else:
                idx = _rank_kernel(shifted, binom_table)
            rep_rank_table[idx] = 0
            rep_sym_table[idx] = g_idx
            rep_amp_table[idx] = np.complex64(alpha)
    else:
        byte_tables = _make_byte_tables(n_site, perm_bits, perm_phases)
        perm_tbl, angle_tbl, n_bytes = byte_tables

        for start in range(0, n_total, chunk):
            stop = min(start + chunk, n_total)
            b = stop - start
            ranks = np.arange(start, stop, dtype=np.int64)
            masks = _np_unrank_chunk(ranks, n_site, n_filled, binom_table)

            # pass 1: min image over g (vectorized)
            min_img = np.full(b, np.iinfo(np.uint64).max, dtype=np.uint64)
            for g_idx in range(n_g):
                img = np.zeros(b, dtype=np.uint64)
                for bb in range(n_bytes):
                    by = ((masks >> (8 * bb)) & 0xFF).astype(np.int64)
                    img |= np.take(perm_tbl[g_idx, bb], by)
                min_img = np.minimum(min_img, img)
            is_rep = min_img == masks
            rep_pos = np.nonzero(is_rep)[0]
            if rep_pos.size == 0:
                continue
            rep_idxs = np.arange(len(repr_list), len(repr_list) + rep_pos.size,
                                 dtype=np.int64)

            # pass 2: images+phases of the reps only (n_rep << B)
            r_masks = masks[rep_pos]
            r_images = np.zeros((n_g, rep_pos.size), dtype=np.uint64)
            r_angles = np.zeros((n_g, rep_pos.size), dtype=np.float64)
            for g_idx in range(n_g):
                for bb in range(n_bytes):
                    by = ((r_masks >> (8 * bb)) & 0xFF).astype(np.int64)
                    r_images[g_idx] |= np.take(perm_tbl[g_idx, bb], by)
                    r_angles[g_idx] += np.take(angle_tbl[g_idx, bb], by)
            r_phases = np.exp(1j * r_angles)
            if stats_flag:
                bits = ((r_masks[:, None] >> np.arange(n_site, dtype=np.uint64))
                        & 1).astype(np.int64)
                for g_idx in range(n_g):
                    inv = bits @ inv_table[g_idx]
                    n_inv = (bits * inv).sum(axis=1)
                    flip = (n_inv & 1) == 1
                    r_phases[g_idx, flip] *= -1.0 + 0.0j

            # scatter (rank the (n_g, n_rep) image block at once)
            rk_all = _np_rank_any(r_images, lin_left, lin_right, lin_n_right,
                                  binom_table, n_site)
            for g_idx in range(n_g):
                target = rk_all[g_idx]
                rep_rank_table[target] = rep_idxs
                rep_sym_table[target] = g_idx
                rep_amp_table[target] = np.complex64(r_phases[g_idx])

            # stabilizer of each rep (group elements that fix it)
            fix = r_images == r_masks[None, :]          # (n_g, n_rep)
            for j in range(rep_pos.size):
                gidx_stab = np.nonzero(fix[:, j])[0]
                repr_list.append(int(r_masks[j]))
                stab_order_list.append(len(gidx_stab))
                stab_gidx_list.append(gidx_stab.astype(np.int64))
                stab_phase_list.append(
                    r_phases[gidx_stab, j].astype(np.complex128)
                )

    repr_arr = np.asarray(repr_list, dtype=np.uint64)
    stab_arr = np.asarray(stab_order_list, dtype=np.int64)
    return repr_arr, stab_arr, stab_gidx_list, stab_phase_list


# ---------------------------------------------------------------------------
# Sparse-block construction kernel (matrix mode)
# ---------------------------------------------------------------------------

def _grow_buffers(Is, Js, Vs, cap):
    """Double the capacity of the sparse-triplet buffers (kept for parity;
    the numpy builders preallocate exactly)."""
    new_cap = cap * 2
    Is_new = np.empty(new_cap, np.int64)
    Is_new[:cap] = Is
    Js_new = np.empty(new_cap, np.int64)
    Js_new[:cap] = Js
    Vs_new = np.empty(new_cap, np.complex128)
    Vs_new[:cap] = Vs
    return Is_new, Js_new, Vs_new, new_cap


def _vectorized_block_triplets(
    repr_masks: np.ndarray,
    stab_orders: np.ndarray,
    i_from_arr: np.ndarray,
    i_to_arr: np.ndarray,
    t_arr: np.ndarray,
    V_i_arr: np.ndarray,
    V_j_arr: np.ndarray,
    V_arr: np.ndarray,
    rep_rank_table: np.ndarray,
    rep_sym_table: np.ndarray,
    rep_amp_table: np.ndarray,
    catalog_repr_masks: np.ndarray,
    binom_table: np.ndarray,
    lin_left: np.ndarray,
    lin_right: np.ndarray,
    lin_n_right: np.int64,
    stats_flag: bool,
    irrep_values: np.ndarray,
    in_basis_mask: np.ndarray,
    rep_to_basis: np.ndarray,
    include_diagonal: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized (row, col, amp) + h_diag construction.

    Shared by the matrix-mode H-block and the matrix-free projection table:
    both store exactly the same hopping triplets; the two callers differ
    only in what they do with them (sparse matrix vs flat gather/scatter
    table).  Each hopping term is processed for ALL columns at once.

    ``include_diagonal``: the matrix-mode caller includes the density
    diagonal as explicit ``(col, col)`` triplets (matching the Julia
    ``build_ed_Hamiltonian_symmetry_block``); the matrix-free caller keeps
    the diagonal only in ``h_diag`` (matching the Julia projection table),
    because its matvec already adds ``h_diag * x`` elementwise.
    """
    sector_dim = repr_masks.shape[0]
    n_bilin = i_from_arr.shape[0]
    lin_n_right = int(lin_n_right)

    hop_masks = np.bitwise_or(
        np.left_shift(np.uint64(1), (i_from_arr - 1).astype(np.uint64)),
        np.left_shift(np.uint64(1), (i_to_arr - 1).astype(np.uint64)))
    from_masks = np.left_shift(np.uint64(1), (i_from_arr - 1).astype(np.uint64))

    # diagonal (density-density) part — vectorized
    h_diag = np.zeros(sector_dim, dtype=np.complex128)
    for k in range(V_i_arr.shape[0]):
        occ_i = (repr_masks & np.left_shift(np.uint64(1), np.uint64(V_i_arr[k] - 1))) != 0
        occ_j = (repr_masks & np.left_shift(np.uint64(1), np.uint64(V_j_arr[k] - 1))) != 0
        both = occ_i & occ_j
        h_diag[both] += V_arr[k]

    if include_diagonal:
        rows = [np.arange(sector_dim, dtype=np.int64)]
        cols = [np.arange(sector_dim, dtype=np.int64)]
        vals = [h_diag]
    else:
        rows, cols, vals = [], [], []

    for k in range(n_bilin):
        gate = (repr_masks & hop_masks[k]) == from_masks[k]
        col_ids = np.nonzero(gate)[0]
        if col_ids.size == 0:
            continue
        new_masks = repr_masks[col_ids] ^ hop_masks[k]
        if lin_n_right >= 0:
            lo_mask = (np.uint64(1) << lin_n_right) - 1
            idx = lin_left[new_masks >> lin_n_right] + lin_right[new_masks & lo_mask]
        else:
            idx = _np_rank_fallback(new_masks, binom_table, 0)
        cat_rep = rep_rank_table[idx]
        in_basis = in_basis_mask[cat_rep]
        if not in_basis.any():
            continue
        keep = np.nonzero(in_basis)[0]
        row_ids = rep_to_basis[cat_rep[keep]]
        col_ids_k = col_ids[keep]
        gk = rep_sym_table[idx[keep]]
        ak = rep_amp_table[idx[keep]]
        hop_phase = np.ones(len(keep), dtype=np.complex128)
        if stats_flag:
            lo = min(int(i_from_arr[k]), int(i_to_arr[k]))
            hi = max(int(i_from_arr[k]), int(i_to_arr[k]))
            between = 0
            if hi - lo > 1:
                between = ((1 << (hi - lo - 1)) - 1) << lo
            flip = (np.bitwise_count(repr_masks[col_ids_k] & np.uint64(between)) & 1) == 1
            hop_phase[flip] = -1.0 + 0.0j
        stab_col = stab_orders[col_ids_k]
        stab_row = stab_orders[row_ids]
        H_elem = (t_arr[k] * hop_phase * ak * np.conj(irrep_values[gk])
                  * np.sqrt(stab_row / stab_col))
        rows.append(row_ids)
        cols.append(col_ids_k)
        vals.append(H_elem)

    return (np.concatenate(rows), np.concatenate(cols),
            np.concatenate(vals), h_diag)


def _build_H_block_kernel(
    repr_masks: np.ndarray,
    stab_orders: np.ndarray,
    i_from_arr: np.ndarray,
    i_to_arr: np.ndarray,
    t_arr: np.ndarray,
    V_i_arr: np.ndarray,
    V_j_arr: np.ndarray,
    V_arr: np.ndarray,
    rep_rank_table: np.ndarray,
    rep_sym_table: np.ndarray,
    rep_amp_table: np.ndarray,
    catalog_repr_masks: np.ndarray,
    binom_table: np.ndarray,
    lin_left: np.ndarray,
    lin_right: np.ndarray,
    lin_n_right: np.int64,
    stats_flag: bool,
    irrep_values: np.ndarray,
    basis_map: Any,
    est_nnz_per_col: np.int64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.int64]:
    """Build the sparse symmetry block (vectorized numpy kernel)."""
    n_cat = len(catalog_repr_masks)
    in_basis_mask = np.zeros(n_cat, dtype=bool)
    rep_to_basis = np.full(n_cat, -1, dtype=np.int64)
    for j in range(n_cat):
        mj = int(catalog_repr_masks[j])
        if mj in basis_map:
            in_basis_mask[j] = True
            rep_to_basis[j] = basis_map[mj]

    Is, Js, Vs, _ = _vectorized_block_triplets(
        repr_masks, stab_orders, i_from_arr, i_to_arr, t_arr,
        V_i_arr, V_j_arr, V_arr, rep_rank_table, rep_sym_table,
        rep_amp_table, catalog_repr_masks, binom_table, lin_left,
        lin_right, lin_n_right, stats_flag, irrep_values,
        in_basis_mask, rep_to_basis,
        include_diagonal=True,
    )
    return Is, Js, Vs, np.int64(len(Is))


# ---------------------------------------------------------------------------
# Matrix-free projection table (implementation difference from the Julia
# source, documented): Julia's matrix-free mode re-projects every scattered
# mask on EVERY matvec through its `Dict` cache.  Numba's typed.Dict was
# ~5× slower per probe than Julia's Dict, which made the Python matvec an
# order of magnitude too slow.  The Python port therefore precomputes the
# projection triplets (row, column, amplitude) ONCE at populate time —
# the same information content as the Julia cache (and less memory than
# Julia's Dict), queried at zero cost during the Lanczos iterations.
# ---------------------------------------------------------------------------


def _build_matrixfree_table_kernel(
    repr_masks: np.ndarray,
    stab_orders: np.ndarray,
    i_from_arr: np.ndarray,
    i_to_arr: np.ndarray,
    t_arr: np.ndarray,
    V_i_arr: np.ndarray,
    V_j_arr: np.ndarray,
    V_arr: np.ndarray,
    rep_rank_table: np.ndarray,
    rep_sym_table: np.ndarray,
    rep_amp_table: np.ndarray,
    catalog_repr_masks: np.ndarray,
    binom_table: np.ndarray,
    lin_left: np.ndarray,
    lin_right: np.ndarray,
    lin_n_right: np.int64,
    stats_flag: bool,
    irrep_values: np.ndarray,
    basis_map: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the matrix-free projection table (vectorized numpy kernel).

    For every representative column and every valid hopping move, compute
    the projected amplitude :math:`t\\,s_{\\mathrm{JW}}\\,\\alpha\\,\\chi(g)^*
    \\sqrt{|\\mathrm{Stab}(row)|/|\\mathrm{Stab}(col)|}` once and store the
    triplet ``(row, col, amplitude)``.  The diagonal density contribution
    is stored per column.

    Returns:
        tuple: ``(row_ind, col_ind, vals, h_diag)``.
    """
    n_cat = len(catalog_repr_masks)
    in_basis_mask = np.zeros(n_cat, dtype=bool)
    rep_to_basis = np.full(n_cat, -1, dtype=np.int64)
    for j in range(n_cat):
        mj = int(catalog_repr_masks[j])
        if mj in basis_map:
            in_basis_mask[j] = True
            rep_to_basis[j] = basis_map[mj]

    return _vectorized_block_triplets(
        repr_masks, stab_orders, i_from_arr, i_to_arr, t_arr,
        V_i_arr, V_j_arr, V_arr, rep_rank_table, rep_sym_table,
        rep_amp_table, catalog_repr_masks, binom_table, lin_left,
        lin_right, lin_n_right, stats_flag, irrep_values,
        in_basis_mask, rep_to_basis,
        include_diagonal=False,
    )


def _apply_table_kernel(
    y_threads: np.ndarray,
    x: np.ndarray,
    row_ind: np.ndarray,
    col_ind: np.ndarray,
    vals: np.ndarray,
    h_diag: np.ndarray,
) -> np.ndarray:
    """Compute ``y = H·x`` from the precomputed projection table.

    Vectorized scatter: the diagonal acts elementwise, the off-diagonal
    triplets are scattered with ``np.bincount`` (C-speed, handles the
    duplicate row indices).  ``y_threads`` is accepted for API parity with
    the old Numba kernel and ignored.
    """
    n = x.shape[0]
    y = np.zeros(n, dtype=np.complex128)
    y[:] = h_diag * x
    contrib = vals * x[col_ind]
    y += np.bincount(row_ind, weights=contrib.real, minlength=n)
    y += 1j * np.bincount(row_ind, weights=contrib.imag, minlength=n)
    return y


def _apply_hamiltonian_kernel(
    y_threads: np.ndarray,
    x: np.ndarray,
    repr_masks: np.ndarray,
    stab_orders: np.ndarray,
    i_from_arr: np.ndarray,
    i_to_arr: np.ndarray,
    t_arr: np.ndarray,
    V_i_arr: np.ndarray,
    V_j_arr: np.ndarray,
    V_arr: np.ndarray,
    rep_rank_table: np.ndarray,
    rep_sym_table: np.ndarray,
    rep_amp_table: np.ndarray,
    catalog_repr_masks: np.ndarray,
    binom_table: np.ndarray,
    lin_left: np.ndarray,
    lin_right: np.ndarray,
    lin_n_right: np.int64,
    stats_flag: bool,
    irrep_values: np.ndarray,
    basis_map: Any,
) -> np.ndarray:
    """Compute ``y = H·x`` on-the-fly (vectorized numpy kernel; no sparse
    matrix, no precomputed table — the canonical projections are computed
    per call from the representative tables).

    Mirrors ``apply_hamiltonian!``: each hopping term is processed for all
    columns at once (gather → canonicalize → scatter).  ``y_threads`` is
    accepted for API parity and ignored (numpy vectorization replaces the
    per-thread buffers).
    """
    n = x.shape[0]
    n_bilin = i_from_arr.shape[0]
    lin_n_right = int(lin_n_right)

    n_cat = len(catalog_repr_masks)
    in_basis_mask = np.zeros(n_cat, dtype=bool)
    rep_to_basis = np.full(n_cat, -1, dtype=np.int64)
    for j in range(n_cat):
        mj = int(catalog_repr_masks[j])
        if mj in basis_map:
            in_basis_mask[j] = True
            rep_to_basis[j] = basis_map[mj]

    hop_masks = np.bitwise_or(
        np.left_shift(np.uint64(1), (i_from_arr - 1).astype(np.uint64)),
        np.left_shift(np.uint64(1), (i_to_arr - 1).astype(np.uint64)))
    from_masks = np.left_shift(np.uint64(1), (i_from_arr - 1).astype(np.uint64))

    y = np.zeros(n, dtype=np.complex128)

    # diagonal
    for k in range(V_i_arr.shape[0]):
        occ_i = (repr_masks & np.left_shift(np.uint64(1), np.uint64(V_i_arr[k] - 1))) != 0
        occ_j = (repr_masks & np.left_shift(np.uint64(1), np.uint64(V_j_arr[k] - 1))) != 0
        both = occ_i & occ_j
        y[both] += V_arr[k] * x[both]

    # off-diagonal
    for k in range(n_bilin):
        gate = (repr_masks & hop_masks[k]) == from_masks[k]
        col_ids = np.nonzero(gate)[0]
        if col_ids.size == 0:
            continue
        new_masks = repr_masks[col_ids] ^ hop_masks[k]
        if lin_n_right >= 0:
            lo_mask = (np.uint64(1) << lin_n_right) - 1
            idx = lin_left[new_masks >> lin_n_right] + lin_right[new_masks & lo_mask]
        else:
            idx = _np_rank_fallback(new_masks, binom_table, 0)
        cat_rep = rep_rank_table[idx]
        in_basis = in_basis_mask[cat_rep]
        if not in_basis.any():
            continue
        keep = np.nonzero(in_basis)[0]
        row_ids = rep_to_basis[cat_rep[keep]]
        col_ids_k = col_ids[keep]
        gk = rep_sym_table[idx[keep]]
        ak = rep_amp_table[idx[keep]]
        hop_phase = np.ones(len(keep), dtype=np.complex128)
        if stats_flag:
            lo = min(int(i_from_arr[k]), int(i_to_arr[k]))
            hi = max(int(i_from_arr[k]), int(i_to_arr[k]))
            between = 0
            if hi - lo > 1:
                between = ((1 << (hi - lo - 1)) - 1) << lo
            flip = (np.bitwise_count(repr_masks[col_ids_k] & np.uint64(between)) & 1) == 1
            hop_phase[flip] = -1.0 + 0.0j
        stab_col = stab_orders[col_ids_k]
        stab_row = stab_orders[row_ids]
        H_elem = (t_arr[k] * hop_phase * ak * np.conj(irrep_values[gk])
                  * np.sqrt(stab_row / stab_col))
        contrib = H_elem * x[col_ids_k]
        y += np.bincount(row_ids, weights=contrib.real, minlength=n)
        y += 1j * np.bincount(row_ids, weights=contrib.imag, minlength=n)

    return y

# ═════════════════════════════════════════════════════════════════════════════
# 1. Symmetry_Operation — a single group element
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(init=False)
class Symmetry_Operation:
    """One single group action on the occupation basis.

    Julia counterpart: ``struct Symmetry_Operation{Group_Label}``.  The action
    on a Fock state is (for both bosons and fermions)

    .. math::
        U_g |n_1,\\dots,n_N\\rangle =
        \\Big(\\prod_{i\\in\\mathrm{occ}} \\eta_g(i)\\Big)\\,\\mathrm{sgn}_g(\\mathrm{occ})\\,
        |n_{\\pi(1)},\\dots,n_{\\pi(N)}\\rangle,

    and in second-quantized form it acts on creation operators as
    :math:`U_g a_i^\\dagger U_g^{-1} = \\eta_g(i)\\,a_{\\pi(i)}^\\dagger`.

    Attributes:
        label: the generic ``Group_Label``-typed label of the group element,
            depending on the symmetry transformation under consideration
            (e.g. a ``(dx, dy)`` tuple for lattice translations).
        perm: the permutation of the whole vertices — a list of length
            :math:`N` whose entries are integers from 1 to :math:`N`
            (1-based).
        perm_phases: the associated per-vertex U(1) phase factor
            :math:`\\eta_g(i)` — a complex array of length :math:`N`.
    """

    label: Any
    perm: list[int]
    perm_phases: np.ndarray

    def __init__(self, label: Any, perm: list[int], *, perm_phases=None):
        n_site = len(perm)
        if perm_phases is None:
            phases = np.full(n_site, COMPLEX_ONE, dtype=np.complex128)
        else:
            phases = np.asarray(perm_phases, dtype=np.complex128)
        assert phases.shape[0] == n_site, (
            "Symmetry_Operation: perm_phases must have length n_site."
        )
        self.label = label
        self.perm = [int(p) for p in perm]
        self.perm_phases = phases

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Symmetry_Operation(label={self.label!r}, perm={self.perm})"


def apply_operation_to_mask(
    m: Mask, op: Symmetry_Operation, particle_statistics: Particle_Statistics
) -> tuple[Mask, complex]:
    """Apply :math:`g \\in G` to a Fock state mask.

    Args:
        m: occupation bitmask of the Fock state.
        op: the symmetry operation :math:`g` to apply.
        particle_statistics: :attr:`Particle_Statistics.BOSONIC` or
            :attr:`Particle_Statistics.FERMIONIC`.

    Returns:
        tuple[Mask, complex]: ``(new_mask, phase)`` where
        ``phase = ∏ η_g(i)`` for bosons and
        ``phase = ∏ η_g(i) × (-1)^{#inversions}`` for fermions
        (the inversion count is tracked with ``popcount(new_mask >> π(i))``).
    """
    tmp = m
    new_mask = Mask(0)
    phase = COMPLEX_ONE
    parity = 1
    stats_flag = particle_statistics is Particle_Statistics.FERMIONIC
    while tmp != 0:
        lsb = tmp & -tmp
        idx = (lsb.bit_length() - 1) + 1  # 1-based site index
        p = op.perm[idx - 1]
        if stats_flag and (new_mask >> p).bit_count() % 2 == 1:
            parity = -parity
        new_mask |= Mask(1) << (p - 1)
        phase *= op.perm_phases[idx - 1]
        tmp ^= lsb
    return new_mask, phase * parity


# ═════════════════════════════════════════════════════════════════════════════
# 2. Finite_Symmetry_Group
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(init=False)
class Finite_Symmetry_Group:
    """A finite symmetry group as an ordered collection of operations.

    Julia counterpart: ``struct Finite_Symmetry_Group <:
    Abstract_Symmetry_Group``.

    Attributes:
        name: group name (e.g. ``"translations"`` or ``"identity"``).
        n_site: number of vertices in the graph.
        operations: ordered list of :class:`Symmetry_Operation` group
            elements acting on the occupation basis.
        identity_idx: **0-based** index of the identity operation in
            ``operations`` (the Julia package uses the 1-based default 1).
    """

    name: str
    n_site: int
    operations: list[Symmetry_Operation]
    identity_idx: int

    def __init__(
        self,
        name: str,
        ops: list[Symmetry_Operation],
        *,
        identity_idx: int = 0,
    ):
        if len(ops) == 0:
            raise ValueError("Finite_Symmetry_Group: ops must not be empty.")
        n_site = len(ops[0].perm)
        for op in ops:
            if len(op.perm) != n_site or op.perm_phases.shape[0] != n_site:
                raise ValueError(
                    "Finite_Symmetry_Group: all operations must have "
                    "perm/perm_phases of length n_site."
                )
        if ops[identity_idx].perm != list(range(1, n_site + 1)):
            raise ValueError(
                "Finite_Symmetry_Group: the identity operation must have "
                "the trivial permutation."
            )
        self.name = name
        self.n_site = n_site
        self.operations = ops
        self.identity_idx = identity_idx

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Finite_Symmetry_Group(name={self.name!r}, n_site={self.n_site}, "
            f"|G|={len(self.operations)})"
        )


def group_order(G: Finite_Symmetry_Group) -> int:
    """Order (number of elements) of the finite symmetry group ``G``.

    Args:
        G: the symmetry group.

    Returns:
        int: ``len(G.operations)``.
    """
    return len(G.operations)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Canonical representative & 4. Gosper's hack (public wrappers)
# ═════════════════════════════════════════════════════════════════════════════


def get_canonical_representative(
    m: Mask, G: Finite_Symmetry_Group, stats: Particle_Statistics
) -> tuple[Mask, int, complex]:
    """Minimal orbit representative of mask ``m`` under group ``G``.

    Julia counterpart: ``get_canonical_representative`` (O(|G|) — used only
    during precomputation and as a fallback).

    Args:
        m: occupation bitmask of the Fock state.
        G: the finite symmetry group.
        stats: particle statistics of the model.

    Returns:
        tuple[Mask, int, complex]: ``(repr, best_g, best_amp)`` — the
        smallest mask in the orbit, the **0-based** index of the group
        element realizing it, and the accumulated U(1) phase.
    """
    repr_mask = m
    best_g = G.identity_idx
    best_amp = COMPLEX_ONE
    for g_idx, g in enumerate(G.operations):
        shifted, alpha = apply_operation_to_mask(m, g, stats)
        if shifted < repr_mask:
            repr_mask = shifted
            best_g = g_idx
            best_amp = alpha
    return repr_mask, best_g, best_amp


def _gosper_next(x: Mask) -> Mask:
    """Next mask with the same popcount after ``x`` (Gosper's hack).

    Args:
        x: current fixed-popcount mask.

    Returns:
        Mask: the next mask in lexicographic order, or a value ≥ ``1 <<
        n_site`` after the last one.
    """
    c = x & -x
    r = x + c
    return (((r ^ x) >> 2) // c) | r


def _first_combination_mask(n_filled: int) -> Mask:
    """Mask with the ``n_filled`` lowest bits set.

    Args:
        n_filled: number of set bits.

    Returns:
        Mask: ``(1 << n_filled) - 1`` (or ``0`` for ``n_filled == 0``).
    """
    if n_filled == 0:
        return Mask(0)
    return (Mask(1) << n_filled) - 1


# ═════════════════════════════════════════════════════════════════════════════
# 6. Symmetry_Orbit_Catalog
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class Symmetry_Orbit_Catalog:
    """Catalog of symmetry orbits at fixed particle number.

    Julia counterpart: ``mutable struct Symmetry_Orbit_Catalog``.

    Attributes:
        symmetry_group: the finite symmetry group used for the decomposition.
        representative_mask_list: canonical representative mask of each orbit.
        stabilizer_order_list: order :math:`|\\mathrm{Stab}|` of the
            stabilizer subgroup of each representative.
        stabilizer_g_indices_list: for each orbit, the **0-based** indices
            of the group elements in the stabilizer.
        stabilizer_phases_list: for each orbit, the accumulated U(1) phases
            :math:`\\alpha_h([\\mathbf{s}])` of the stabilizer elements.
        rep_rank_table: XDiag-style dense table over the FULL fixed-filling
            basis, indexed by the combinadic rank: the **0-based**
            representative index of the orbit containing that mask
            (``-1`` = unset).
        rep_sym_table: the **0-based** group-element index ``g`` with
            ``g(rep) = mask``.
        rep_amp_table: the canonicalization amplitude ``α_g(rep)``
            (``complex64``; the per-site U(1) phases make it
            state-dependent, unlike XDiag's pure-permutation case).
        binom_table: the precomputed binomial table used by
            :func:`rank_combination` (``(n_site+1) × (n_filled+1)``).
    """

    symmetry_group: Finite_Symmetry_Group
    representative_mask_list: list[Mask]
    stabilizer_order_list: list[int]
    stabilizer_g_indices_list: list[list[int]]
    stabilizer_phases_list: list[list[complex]]
    rep_rank_table: np.ndarray
    rep_sym_table: np.ndarray
    rep_amp_table: np.ndarray
    binom_table: np.ndarray
    lin_left: np.ndarray
    lin_right: np.ndarray
    lin_n_right: int


def rank_combination(m: Mask, binom_table: np.ndarray) -> int:
    """Combinadic (colexicographic) rank of a fixed-popcount bitmask.

    Public counterpart of the Numba kernel ``_rank_kernel`` (the same
    formula, for ordinary Python integers): a bijection between k-subsets
    of ``1..n_site`` and ``[0, C(n_site, k))`` needing no hash table.

    Args:
        m: the occupation bitmask.
        binom_table: precomputed ``(n_site+1) × (n_filled+1)`` binomial
            table (from :func:`_binom_table`).

    Returns:
        int: the colexicographic rank of ``m``.
    """
    idx = 0
    j = 1
    tmp = m
    while tmp != 0:
        lsb = tmp & -tmp
        b = lsb.bit_length() - 1  # 0-based bit position
        idx += int(binom_table[b, j])
        j += 1
        tmp ^= lsb
    return idx


def update_orbit_stabilizer_phases(
    catalog: Symmetry_Orbit_Catalog,
    new_group: Finite_Symmetry_Group,
    particle_statistics: Particle_Statistics,
) -> Symmetry_Orbit_Catalog:
    """**In-place** update of stabilizer phases for a gauge-covariant group.

    Julia counterpart: ``update_orbit_stabilizer_phases!``.

    When the symmetry group is replaced by a gauge-covariant (e.g.
    flux-aware) copy, the orbit *partition* (representative masks and
    stabilizer group-element indices) depends only on the permutation part
    of the operations, which is unchanged, so only the accumulated U(1)
    phases need recomputation — including the canonicalization amplitudes
    in the representative tables.  This avoids re-running Gosper's hack at
    every flux point.

    Args:
        catalog: the orbit catalog to update in-place.
        new_group: the replacement symmetry group (same permutation part).
        particle_statistics: particle statistics of the model.

    Returns:
        Symmetry_Orbit_Catalog: the same (mutated) catalog.
    """
    if catalog.symmetry_group is new_group:
        return catalog  # no-op
    catalog.symmetry_group = new_group
    for orbit_idx in range(len(catalog.representative_mask_list)):
        gidxs = catalog.stabilizer_g_indices_list[orbit_idx]
        phases = catalog.stabilizer_phases_list[orbit_idx]
        for j, gidx in enumerate(gidxs):
            _, alpha = apply_operation_to_mask(
                catalog.representative_mask_list[orbit_idx],
                new_group.operations[gidx],
                particle_statistics,
            )
            phases[j] = alpha
    # refresh the canonicalization amplitudes in the representative tables
    for orbit_idx in range(len(catalog.representative_mask_list)):
        repr_mask = catalog.representative_mask_list[orbit_idx]
        for gidx in range(group_order(new_group)):
            shifted, alpha = apply_operation_to_mask(
                repr_mask, new_group.operations[gidx], particle_statistics
            )
            idx = rank_combination(shifted, catalog.binom_table)
            if catalog.rep_rank_table[idx] == orbit_idx:
                catalog.rep_sym_table[idx] = gidx
                catalog.rep_amp_table[idx] = np.complex64(alpha)
    return catalog


def _group_kernel_arrays(
    G: Finite_Symmetry_Group,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute per-group kernel arrays: perm_bits, perms, perm_phases.

    Args:
        G: the symmetry group.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(perm_bits, perms,
        perm_phases)`` of shapes ``(nG, n_site)`` — the pre-shifted bit for
        each ``(g, i)`` pair, the permutation index, and the U(1) phase.
    """
    n_g = group_order(G)
    n_site = G.n_site
    perm_bits = np.empty((n_g, n_site), dtype=np.uint64)
    perms = np.empty((n_g, n_site), dtype=np.int64)
    perm_phases = np.empty((n_g, n_site), dtype=np.complex128)
    for g_idx, op in enumerate(G.operations):
        for i in range(n_site):
            perm_bits[g_idx, i] = np.uint64(1) << (op.perm[i] - 1)
            perms[g_idx, i] = op.perm[i]
            perm_phases[g_idx, i] = op.perm_phases[i]
    return perm_bits, perms, perm_phases


def build_symmetry_orbit_catalog(
    *,
    second_quantized_model: Real_Space_Second_Quantized_Model,
    n_filled: int,
    symmetry_group: Finite_Symmetry_Group,
    particle_statistics: Particle_Statistics,
) -> Symmetry_Orbit_Catalog:
    """Partition the fixed-filling Hilbert space into symmetry orbits.

    Julia counterpart: ``build_symmetry_orbit_catalog``.  Gosper's hack
    enumerates all :math:`\\binom{N}{N_e}` masks in lexicographic order;
    a mask is a representative iff it is the minimum of its orbit
    (XDiag's ``isrepresentative`` early-exit test — no ``seen`` set), and
    each representative's orbit is expanded in the same pass into the
    dense representative tables (indexed by the combinadic rank).

    Args:
        second_quantized_model: the model (used for ``lattice.n_site``).
        n_filled: number of filled particles (set bits).
        symmetry_group: the symmetry group.
        particle_statistics: particle statistics of the model.

    Returns:
        Symmetry_Orbit_Catalog: the orbit catalog.
    """
    n_site = symmetry_group.n_site
    assert n_site == second_quantized_model.lattice.n_site, (
        "build_symmetry_orbit_catalog: symmetry_group.n_site must equal "
        "lattice.n_site."
    )
    if n_site > MAX_N_SITE_UINT64:
        raise ValueError(
            f"n_site = {n_site} exceeds the 64-bit kernel limit "
            f"({MAX_N_SITE_UINT64}); use a smaller system."
        )

    n_total = math.comb(n_site, n_filled)
    n_orbits_est = math.ceil(n_total / group_order(symmetry_group))

    n_g = group_order(symmetry_group)
    assert 0 <= n_filled <= n_site

    print(
        f"\tBuilding symmetry-orbit catalog (n_filled={n_filled}, "
        f"|G|={n_g}) ... ",
        end="",
        flush=True,
    )

    import time

    t0 = time.perf_counter()
    perm_bits, perms, perm_phases = _group_kernel_arrays(symmetry_group)
    stats_flag = particle_statistics is Particle_Statistics.FERMIONIC
    binom_table = _binom_table(n_site, n_filled)
    if n_site <= LIN_TABLE_MAX_N_SITE:
        lin_left, lin_right, lin_n_right = _lin_table(n_site, n_filled)
    else:
        lin_left = np.zeros(1, dtype=np.int64)
        lin_right = np.zeros(1, dtype=np.int64)
        lin_n_right = -1
    rep_rank_table = np.full(n_total, -1, dtype=np.int32)
    rep_sym_table = np.zeros(n_total, dtype=np.int32)
    rep_amp_table = np.zeros(n_total, dtype=np.complex64)
    repr_arr, stab_arr, stab_gidx_tl, stab_phase_tl = _build_orbit_catalog_kernel(
        np.int64(n_site),
        np.int64(n_filled),
        perm_bits,
        perms,
        perm_phases,
        stats_flag,
        binom_table,
        lin_left,
        lin_right,
        np.int64(lin_n_right),
        rep_rank_table,
        rep_sym_table,
        rep_amp_table,
    )
    elapsed = time.perf_counter() - t0

    repr_list = [int(m) for m in repr_arr]
    stab_order_list = [int(v) for v in stab_arr]
    stab_gidx_list = [[int(g) for g in arr] for arr in stab_gidx_tl]
    stab_phase_list = [
        [complex(v) for v in arr] for arr in stab_phase_tl
    ]

    n_orbits = len(repr_list)
    print(
        f"Done. {n_orbits} orbits (reduction "
        f"{round(n_orbits / n_total * 100, 1)}%).  t={round(elapsed, 3)}s"
    )
    return Symmetry_Orbit_Catalog(
        symmetry_group,
        repr_list,
        stab_order_list,
        stab_gidx_list,
        stab_phase_list,
        rep_rank_table,
        rep_sym_table,
        rep_amp_table,
        binom_table,
        lin_left,
        lin_right,
        lin_n_right,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 7. OneDim_Irrep
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(init=False)
class OneDim_Irrep:
    """One single one-dimensional irreducible representation (irrep).

    Julia counterpart: ``struct OneDim_Irrep{Irrep_Label}``, labelled by a
    generic-typed ``label``.

    Attributes:
        label: the ``Irrep_Label``-typed label of the irrep; can be a
            string, integer, tuple, etc., depending on the symmetry under
            consideration (e.g. momentum ``(k1, k2)`` tuples).
        values: the irrep values for each group element — a complex array
            of length ``|G|`` whose ``g``-th entry is the character
            :math:`\\chi(g)` of the ``g``-th group operation.
    """

    label: Any
    values: np.ndarray

    def __init__(self, label: Any, values):
        self.label = label
        self.values = np.asarray(values, dtype=np.complex128)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"OneDim_Irrep(label={self.label!r}, |values|={len(self.values)})"


# ═════════════════════════════════════════════════════════════════════════════
# 8. Symmetry_Sector_Basis
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class Symmetry_Sector_Basis:
    """Symmetry-sector basis for a given 1-D irrep.

    Julia counterpart: ``struct Symmetry_Sector_Basis``.

    Attributes:
        irrep: the 1-D irrep for which this sector basis is constructed.
        symmetry_group: the symmetry group under consideration.
        representative_mask_list: representative masks of the orbits
            belonging to this irrep/sector, inherited from the
            :class:`Symmetry_Orbit_Catalog`.
        stabilizer_order_list: stabilizer-group order for each
            representative mask in this sector.
        representative_mask_to_mask_idx_map: ``dict[mask → idx]`` mapping a
            representative mask to its **0-based** index in
            ``representative_mask_list``.
    """

    irrep: OneDim_Irrep
    symmetry_group: Finite_Symmetry_Group
    representative_mask_list: list[Mask]
    stabilizer_order_list: list[int]
    representative_mask_to_mask_idx_map: dict[Mask, int]


def _basis_index(basis: Symmetry_Sector_Basis, repr_mask: Mask) -> int | None:
    """Look up the sector-basis index of a representative mask.

    Args:
        basis: the sector basis.
        repr_mask: a canonical representative mask.

    Returns:
        int | None: the **0-based** basis index, or ``None`` if the orbit
        does not belong to this sector (Julia returns the sentinel ``0``).
    """
    return basis.representative_mask_to_mask_idx_map.get(repr_mask)


def _is_orbit_compatible(
    catalog: Symmetry_Orbit_Catalog,
    orbit_idx: int,
    irrep: OneDim_Irrep,
    *,
    atol: float = 1e-12,
) -> bool:
    """Whether the orbit's stabilizer phases match the irrep character.

    An orbit :math:`[\\mathbf{s}]` contributes to irrep :math:`\\chi` iff

    .. math::
        \\chi(h) = \\alpha_h([\\mathbf{s}]) \\qquad \\forall\\,h\\in\\mathrm{Stab}.

    Args:
        catalog: the orbit catalog.
        orbit_idx: **0-based** orbit index to test.
        irrep: the candidate 1-D irrep.
        atol: absolute tolerance of the phase comparison.

    Returns:
        bool: ``True`` if the orbit is compatible with the irrep.
    """
    gidxs = catalog.stabilizer_g_indices_list[orbit_idx]
    phases = catalog.stabilizer_phases_list[orbit_idx]
    for j in range(len(gidxs)):
        if not abs(irrep.values[gidxs[j]] - phases[j]) <= atol:
            return False
    return True


def build_symmetry_sector_basis(
    catalog: Symmetry_Orbit_Catalog, irrep: OneDim_Irrep, *, atol: float = 1e-12
) -> Symmetry_Sector_Basis:
    """Filter the orbit catalog down to the orbits of one irrep sector.

    Julia counterpart: ``build_symmetry_sector_basis``.

    Args:
        catalog: the orbit catalog.
        irrep: the target 1-D irrep.
        atol: absolute tolerance of the stabilizer-phase compatibility
            check.

    Returns:
        Symmetry_Sector_Basis: the sector basis.
    """
    repr_list: list[Mask] = []
    stab_order_list: list[int] = []
    for i in range(len(catalog.representative_mask_list)):
        if _is_orbit_compatible(catalog, i, irrep, atol=atol):
            repr_list.append(catalog.representative_mask_list[i])
            stab_order_list.append(catalog.stabilizer_order_list[i])
    representative_mask_to_mask_idx_map = {
        m: idx for idx, m in enumerate(repr_list)
    }
    return Symmetry_Sector_Basis(
        irrep,
        catalog.symmetry_group,
        repr_list,
        stab_order_list,
        representative_mask_to_mask_idx_map,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 9. CanonicalMap — O(1) canonical-representative lookup (all modes)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(init=False)
class CanonicalMap:
    """Precomputed representative lookup: scattered mask → ``(repr, g, α)``.

    Julia counterpart: ``struct CanonicalMap``.  Backed by the XDiag-style
    representative tables built once inside the orbit catalog
    (``rep_rank_table`` / ``rep_sym_table`` / ``rep_amp_table``, indexed
    by the combinadic rank — no lazy Dict, no hashing).  Used uniformly
    across **all** ED modes (matrix, distributed-matrix, and matrix-free)
    to avoid repeated O(|G|) canonicalization; every lookup is a handful
    of array reads.

    Attributes:
        symmetry_group: the finite symmetry group.
        particle_statistics: particle statistics of the model.
        catalog: the orbit catalog whose representative tables back the
            lookups.
    """

    symmetry_group: Finite_Symmetry_Group
    particle_statistics: Particle_Statistics
    catalog: Symmetry_Orbit_Catalog

    def __init__(
        self,
        symmetry_group: Finite_Symmetry_Group,
        particle_statistics: Particle_Statistics,
        catalog: Symmetry_Orbit_Catalog,
    ):
        self.symmetry_group = symmetry_group
        self.particle_statistics = particle_statistics
        self.catalog = catalog
        self._perm_bits, self._perms, self._perm_phases = _group_kernel_arrays(
            symmetry_group
        )
        self._stats_flag = particle_statistics is Particle_Statistics.FERMIONIC

    def __len__(self) -> int:
        return int(np.count_nonzero(self.catalog.rep_rank_table >= 0))


def get_canonical(
    cmap: CanonicalMap, m: Mask
) -> tuple[Mask, int, complex]:
    """O(1) canonical-representative lookup via the representative tables.

    Args:
        cmap: the canonical-map cache.
        m: a scattered occupation mask.

    Returns:
        tuple[Mask, int, complex]: ``(repr, g_idx, α)`` — the canonical
        representative of ``m``, the **0-based** group-element index
        realizing it, and the accumulated phase.
    """
    catalog = cmap.catalog
    if catalog.lin_n_right >= 0:
        idx = int(
            _lin_index_kernel(
                np.uint64(m),
                catalog.lin_left,
                catalog.lin_right,
                np.int64(catalog.lin_n_right),
            )
        )
    else:
        idx = rank_combination(m, catalog.binom_table)
    rep_idx = int(catalog.rep_rank_table[idx])
    if rep_idx < 0:
        # fallback for masks outside the fixed-filling basis
        return get_canonical_representative(
            m, cmap.symmetry_group, cmap.particle_statistics
        )
    repr_mask = catalog.representative_mask_list[rep_idx]
    return (
        repr_mask,
        int(catalog.rep_sym_table[idx]),
        complex(catalog.rep_amp_table[idx]),
    )


def populate_canonical_map(
    cmap: CanonicalMap,
    basis: Symmetry_Sector_Basis,
    bilinear_terms: list[tuple[int, int, complex]],
) -> CanonicalMap:
    """Single-threaded cache warm-up (no-op; tables are prebuilt).

    The representative tables are precomputed at catalog-build time (see
    :func:`build_symmetry_orbit_catalog`); there is no lazy Dict to warm.
    Kept for API compatibility with the previous CanonicalMap design.

    Args:
        cmap: the canonical-map cache.
        basis: the symmetry-sector basis.
        bilinear_terms: the hopping terms ``(i_from, i_to, t)``.

    Returns:
        CanonicalMap: the same (unchanged) cache object.
    """
    return cmap


# ═════════════════════════════════════════════════════════════════════════════
# 10. project_to_sector — unified irrep projection via CanonicalMap
# ═════════════════════════════════════════════════════════════════════════════


def project_to_sector(
    m: Mask, basis: Symmetry_Sector_Basis, cmap: CanonicalMap
) -> tuple[int, complex] | None:
    """Project a scattered Fock mask onto the symmetry-sector basis.

    Julia counterpart: ``project_to_sector``.  Steps:

    1. obtain the canonical representative ``(repr, g_idx, α)`` via
       :class:`CanonicalMap` (combinadic rank + array reads);
    2. look the representative up in the basis dict;
    3. return ``(row_index, α · χ(g_idx)ˣ)`` or ``None`` if the orbit does
       not belong to this irrep sector.

    Used uniformly by the matrix and matrix-free modes.

    Args:
        m: a scattered occupation mask.
        basis: the symmetry-sector basis.
        cmap: the canonical-map cache.

    Returns:
        tuple[int, complex] | None: ``(row_index, coeff)`` with the
        **0-based** basis row and the projection coefficient, or ``None``.
    """
    repr_mask, g_idx, alpha = get_canonical(cmap, m)
    idx = _basis_index(basis, repr_mask)
    if idx is None:
        return None
    return (idx, alpha * np.conj(basis.irrep.values[g_idx]))


def project_to_unnormalized_sector(
    m: Mask, basis: Symmetry_Sector_Basis, stats: Particle_Statistics = Particle_Statistics.BOSONIC
) -> tuple[int, complex] | None:
    """Legacy wrapper: project a raw mask to the unnormalized sector basis.

    Julia counterpart: ``project_to_unnormalized_sector``.  Unlike
    :func:`project_to_sector`, no cache is used: the canonical
    representative is recomputed from scratch each call.

    Args:
        m: a raw (not necessarily canonical) occupation mask.
        basis: the symmetry-sector basis.
        stats: particle statistics of the model.

    Returns:
        tuple[int, complex] | None: ``(repr_idx, coeff)`` or ``None``.
    """
    repr_mask, g_idx, alpha = get_canonical_representative(
        m, basis.symmetry_group, stats
    )
    idx = _basis_index(basis, repr_mask)
    if idx is None:
        return None
    return (idx, alpha * np.conj(basis.irrep.values[g_idx]))


def build_matrixfree_projection_table(
    basis: Symmetry_Sector_Basis,
    bilinear_terms: list[tuple[int, int, complex]],
    density_terms: list[tuple[int, int, complex]],
    cmap: CanonicalMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the matrix-free projection table of one symmetry sector.

    Julia counterpart: ``build_matrixfree_projection_table``.  Precomputes,
    once per sector, the projected hopping amplitudes
    :math:`t\\,s_{\\mathrm{JW}}\\,\\alpha\\,\\chi(g)^*
    \sqrt{|\mathrm{Stab}(row)|/|\mathrm{Stab}(col)|}` for every valid
    hopping move as flat ``(row, column, amplitude)`` triplets, plus the
    per-column diagonal (density-density) contribution.  Each matrix-free
    ``H·x`` then becomes a pure gather–scatter kernel with zero per-hop
    dict lookups (see :class:`MatrixFreeHamiltonian`).

    Args:
        basis: the symmetry-sector basis.
        bilinear_terms: hopping terms ``(i_from, i_to, t)``.
        density_terms: density-density terms ``(i, j, V)``.
        cmap: canonical-map cache (representative tables).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ``(row_ind, col_ind, vals, h_diag)``.
    """
    sec = _sector_kernel_arrays(basis, cmap)
    i_from_arr, i_to_arr, t_arr = _term_arrays(bilinear_terms)
    V_i_arr, V_j_arr, V_arr = _term_arrays(density_terms)
    return _build_matrixfree_table_kernel(
        sec["repr_masks"],
        sec["stab_orders"],
        i_from_arr,
        i_to_arr,
        t_arr,
        V_i_arr,
        V_j_arr,
        V_arr,
        sec["rep_rank_table"],
        sec["rep_sym_table"],
        sec["rep_amp_table"],
        sec["catalog_repr_masks"],
        sec["binom_table"],
        sec["lin_left"],
        sec["lin_right"],
        sec["lin_n_right"],
        sec["stats_flag"],
        sec["irrep_values"],
        sec["basis_map"],
    )


# ═════════════════════════════════════════════════════════════════════════════
# 11. Matrix-free Hamiltonian application — the core Lanczos operation
# ═════════════════════════════════════════════════════════════════════════════


def hopping_phase_for_stats(
    particle_statistics: Particle_Statistics,
    m: Mask,
    i_from: int,
    i_to: int,
) -> complex:
    """Sign factor of the hop ``i_from → i_to`` in configuration ``m``.

    Julia counterpart: ``hopping_phase_for_stats``.  Bosons always get
    :math:`+1`; fermions get the Jordan–Wigner string phase

    .. math::
       (-1)^{\\#\\{\\text{occupied } j : \\min(i_{\\mathrm{from}},i_{\\mathrm{to}})
       < j < \\max(i_{\\mathrm{from}},i_{\\mathrm{to}})\\}}.

    Args:
        particle_statistics: particle statistics of the model.
        m: occupation mask of the initial configuration.
        i_from: **1-based** source site.
        i_to: **1-based** target site.

    Returns:
        complex: ``+1`` or ``-1``.
    """
    if particle_statistics is not Particle_Statistics.FERMIONIC or i_from == i_to:
        return COMPLEX_ONE
    lo = min(i_from, i_to)
    hi = max(i_from, i_to)
    between = Mask(0)
    if hi - lo > 1:
        between = ((Mask(1) << (hi - lo - 1)) - 1) << lo
    if (m & between).bit_count() % 2 == 1:
        return -COMPLEX_ONE
    return COMPLEX_ONE


def _matrixfree_buffers(n: int) -> np.ndarray:
    """Allocate the accumulation buffer for the matrix-free apply.

    Args:
        n: sector dimension (vector length).

    Returns:
        np.ndarray: ``(1, n)`` complex buffer (kept for API parity with the
        old Numba kernel; the numpy vectorized path needs no per-thread
        buffers).
    """
    return np.zeros((1, n), dtype=np.complex128)


def _term_arrays(
    terms: list[tuple[int, int, complex]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a term list into three numpy arrays (kernel-friendly).

    Args:
        terms: list of ``(i, j, amplitude)`` tuples.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(i_arr, j_arr, amp_arr)``.
    """
    if not terms:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.complex128),
        )
    i_arr = np.asarray([t[0] for t in terms], dtype=np.int64)
    j_arr = np.asarray([t[1] for t in terms], dtype=np.int64)
    amp_arr = np.asarray([complex(t[2]) for t in terms], dtype=np.complex128)
    return i_arr, j_arr, amp_arr


def _sector_kernel_arrays(
    basis: Symmetry_Sector_Basis, cmap: CanonicalMap
) -> dict[str, Any]:
    """Assemble all kernel arrays for a sector basis + canonical map.

    Args:
        basis: the symmetry-sector basis.
        cmap: the canonical-map cache (representative tables).

    Returns:
        dict[str, Any]: numpy arrays and the basis-map dict consumed by the
        numpy kernels.
    """
    basis_map = dict(basis.representative_mask_to_mask_idx_map)
    catalog = cmap.catalog
    return {
        "repr_masks": as_uint64_array(basis.representative_mask_list),
        "stab_orders": np.asarray(basis.stabilizer_order_list, dtype=np.int64),
        "irrep_values": np.asarray(basis.irrep.values, dtype=np.complex128),
        "basis_map": basis_map,
        "rep_rank_table": catalog.rep_rank_table,
        "rep_sym_table": catalog.rep_sym_table,
        "rep_amp_table": catalog.rep_amp_table,
        "catalog_repr_masks": as_uint64_array(
            catalog.representative_mask_list
        ),
        "binom_table": catalog.binom_table,
        "lin_left": catalog.lin_left,
        "lin_right": catalog.lin_right,
        "lin_n_right": np.int64(catalog.lin_n_right),
        "stats_flag": cmap._stats_flag,
    }


@dataclass(init=False)
class MatrixFreeHamiltonian:
    """On-the-fly Hamiltonian operator ``H(x) → y`` (matrix-free mode).

    Julia counterpart: ``struct MatrixFreeHamiltonian``.  No *sparse
    matrix* is ever stored; instead the projected hopping amplitudes are
    precomputed once into a flat projection table (``row, column,
    amplitude`` triplets — the same information as Julia's CanonicalMap
    Dict, in less memory) and each ``H·x`` is a pure gather–scatter
    kernel.  See the module-level note on this documented implementation
    difference (Julia re-projects from its Dict on every matvec, so Python
    precomputes the projection table once and each ``H·x`` is a single
    sparse matvec on the CSR form of that table).

    Attributes:
        basis: the symmetry-sector basis.
        bilinear_terms: hopping terms ``(i_from, i_to, t)``.
        density_terms: density-density terms ``(i, j, V)``.
        cmap: the canonical-map cache (must be pre-populated).
        y_threads: per-thread accumulation buffers of shape
            ``(num_threads, sector_dim)``.
    """

    basis: Symmetry_Sector_Basis
    bilinear_terms: list[tuple[int, int, complex]]
    density_terms: list[tuple[int, int, complex]]
    cmap: CanonicalMap
    y_threads: np.ndarray

    def __init__(
        self,
        basis: Symmetry_Sector_Basis,
        bilinear_terms: list[tuple[int, int, complex]],
        density_terms: list[tuple[int, int, complex]],
        cmap: CanonicalMap,
    ):
        n = len(basis.representative_mask_list)
        self.basis = basis
        self.bilinear_terms = [(i, j, complex(t)) for (i, j, t) in bilinear_terms]
        self.density_terms = [(i, j, complex(v)) for (i, j, v) in density_terms]
        self.cmap = cmap
        self.y_threads = _matrixfree_buffers(n)
        self._sec = _sector_kernel_arrays(basis, cmap)
        self._bi = _term_arrays(self.bilinear_terms)
        self._di = _term_arrays(self.density_terms)
        # Precompute the projection table once (see the kernel docstring for
        # why Python precomputes instead of re-projecting per matvec).
        self._row_ind, self._col_ind, self._table_vals, self._h_diag = (
            _build_matrixfree_table_kernel(
                self._sec["repr_masks"],
                self._sec["stab_orders"],
                self._bi[0],
                self._bi[1],
                self._bi[2],
                self._di[0],
                self._di[1],
                self._di[2],
                self._sec["rep_rank_table"],
                self._sec["rep_sym_table"],
                self._sec["rep_amp_table"],
                self._sec["catalog_repr_masks"],
                self._sec["binom_table"],
                self._sec["lin_left"],
                self._sec["lin_right"],
                self._sec["lin_n_right"],
                self._sec["stats_flag"],
                self._sec["irrep_values"],
                self._sec["basis_map"],
            )
        )
        # Build a single CSR matrix from the projection table: the table is
        # exactly the sector Hamiltonian in coordinate form (off-diagonal
        # triplets + diagonal vector), and one sparse matvec per call is
        # the fastest possible numpy/scipy ``H·x`` (BLAS-level gather).
        self._H = sp.csr_matrix(
            (self._table_vals, (self._row_ind, self._col_ind)),
            shape=(n, n),
        )
        self._H.setdiag(self._H.diagonal() + self._h_diag)
        self._H.eliminate_zeros()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the Hamiltonian: ``y = H·x``.

        Args:
            x: input complex vector of length ``sector_dim``.

        Returns:
            np.ndarray: ``H·x``.
        """
        # One sparse matvec from the precomputed CSR (the projection table
        # is the sector Hamiltonian in coordinate form).
        return self._H @ np.ascontiguousarray(x, dtype=np.complex128)


def apply_hamiltonian(
    y: np.ndarray,
    x: np.ndarray,
    basis: Symmetry_Sector_Basis,
    bilinear_terms: list[tuple[int, int, complex]],
    density_terms: list[tuple[int, int, complex]],
    cmap: CanonicalMap,
) -> np.ndarray:
    """Compute ``y = H·x`` on-the-fly (fresh thread buffers each call).

    Julia counterpart: the 6-argument method of ``apply_hamiltonian!``
    (allocates ``_matrixfree_buffers`` then delegates to the 7-argument
    method).

    Args:
        y: output vector (overwritten with ``H·x``).
        x: input vector.
        basis: the symmetry-sector basis.
        bilinear_terms: hopping terms ``(i_from, i_to, t)``.
        density_terms: density-density terms ``(i, j, V)``.
        cmap: canonical-map cache, pre-populated via
            :func:`populate_canonical_map`.

    Returns:
        np.ndarray: the same array ``y``, now holding ``H·x``.
    """
    y_threads = _matrixfree_buffers(len(x))
    return apply_hamiltonian_with_threads(
        y, x, basis, bilinear_terms, density_terms, cmap, y_threads
    )


def apply_hamiltonian_with_threads(
    y: np.ndarray,
    x: np.ndarray,
    basis: Symmetry_Sector_Basis,
    bilinear_terms: list[tuple[int, int, complex]],
    density_terms: list[tuple[int, int, complex]],
    cmap: CanonicalMap,
    y_threads: np.ndarray,
) -> np.ndarray:
    """Compute ``y = H·x`` on-the-fly with caller-provided thread buffers.

    Julia counterpart: the 7-argument method of ``apply_hamiltonian!``.
    Uses a Numba ``prange`` with per-thread accumulation buffers for
    race-free shared-memory parallelism (equivalent to XDiag's OpenMP
    backend).  The ``cmap`` must be pre-populated before the first call.

    Args:
        y: output vector (overwritten with ``H·x``).
        x: input vector.
        basis: the symmetry-sector basis.
        bilinear_terms: hopping terms ``(i_from, i_to, t)``.
        density_terms: density-density terms ``(i, j, V)``.
        cmap: canonical-map cache (pre-populated).
        y_threads: per-thread buffers of shape ``(num_threads, n)``.

    Returns:
        np.ndarray: the same array ``y``, now holding ``H·x``.
    """
    n = len(x)
    assert len(y) == n

    sec = _sector_kernel_arrays(basis, cmap)
    i_from_arr, i_to_arr, t_arr = _term_arrays(bilinear_terms)
    V_i_arr, V_j_arr, V_arr = _term_arrays(density_terms)

    y[:] = _apply_hamiltonian_kernel(
        y_threads,
        np.ascontiguousarray(x, dtype=np.complex128),
        sec["repr_masks"],
        sec["stab_orders"],
        i_from_arr,
        i_to_arr,
        t_arr,
        V_i_arr,
        V_j_arr,
        V_arr,
        sec["rep_rank_table"],
        sec["rep_sym_table"],
        sec["rep_amp_table"],
        sec["catalog_repr_masks"],
        sec["binom_table"],
        sec["lin_left"],
        sec["lin_right"],
        sec["lin_n_right"],
        sec["stats_flag"],
        sec["irrep_values"],
        sec["basis_map"],
    )
    return y


def hamiltonian_linear_operator(
    basis: Symmetry_Sector_Basis,
    bilinear_terms: list[tuple[int, int, complex]],
    density_terms: list[tuple[int, int, complex]],
    cmap: CanonicalMap,
) -> tuple[MatrixFreeHamiltonian, int]:
    """Return the callable ``H_op(x) → y`` for iterative eigensolvers.

    Julia counterpart: ``hamiltonian_linear_operator`` (the callable is a
    :class:`MatrixFreeHamiltonian` suitable for ARPACK).

    Args:
        basis: the symmetry-sector basis.
        bilinear_terms: hopping terms ``(i_from, i_to, t)``.
        density_terms: density-density terms ``(i, j, V)``.
        cmap: canonical-map cache (pre-populated).

    Returns:
        tuple[MatrixFreeHamiltonian, int]: ``(H_op, n)`` where ``n`` is the
        sector dimension.
    """
    n = len(basis.representative_mask_list)
    return MatrixFreeHamiltonian(basis, bilinear_terms, density_terms, cmap), n


# ═════════════════════════════════════════════════════════════════════════════
# 12. Sparse-matrix construction (matrix mode)
# ═════════════════════════════════════════════════════════════════════════════


def build_ed_Hamiltonian_symmetry_block(
    basis: Symmetry_Sector_Basis,
    bilinear_terms: list[tuple[int, int, complex]],
    density_terms: list[tuple[int, int, complex]],
    cmap: CanonicalMap,
) -> sp.csr_matrix:
    """Build the sparse Hamiltonian block of one symmetry sector.

    Julia counterpart: ``build_ed_Hamiltonian_symmetry_block`` (returns
    ``SparseMatrixCSC``; the Python port returns a CSR matrix, the natural
    scipy analogue).

    Args:
        basis: the symmetry-sector basis.
        bilinear_terms: hopping terms ``(i_from, i_to, t)``.
        density_terms: density-density terms ``(i, j, V)``.
        cmap: canonical-map cache (warms naturally during the loop).

    Returns:
        scipy.sparse.csr_matrix: the ``sector_dim × sector_dim`` sparse
        Hermitian block.
    """
    sector_dim = len(basis.representative_mask_list)
    print(
        f"\tBuilding H block (matrix mode) @ irrep {basis.irrep.label!r} "
        f"(dim={sector_dim}) ... ",
        end="",
        flush=True,
    )

    sec = _sector_kernel_arrays(basis, cmap)
    i_from_arr, i_to_arr, t_arr = _term_arrays(bilinear_terms)
    V_i_arr, V_j_arr, V_arr = _term_arrays(density_terms)
    est_nnz = 1 + 4 * basis.symmetry_group.n_site

    import time

    t0 = time.perf_counter()
    Is, Js, Vs, nnz = _build_H_block_kernel(
        sec["repr_masks"],
        sec["stab_orders"],
        i_from_arr,
        i_to_arr,
        t_arr,
        V_i_arr,
        V_j_arr,
        V_arr,
        sec["rep_rank_table"],
        sec["rep_sym_table"],
        sec["rep_amp_table"],
        sec["catalog_repr_masks"],
        sec["binom_table"],
        sec["lin_left"],
        sec["lin_right"],
        sec["lin_n_right"],
        sec["stats_flag"],
        sec["irrep_values"],
        sec["basis_map"],
        np.int64(est_nnz),
    )
    elapsed = time.perf_counter() - t0

    H = sp.coo_matrix(
        (Vs, (Is, Js)), shape=(sector_dim, sector_dim)
    ).tocsr()
    H.eliminate_zeros()
    nnz_H = H.nnz
    sparsity = 0.0 if sector_dim == 0 else nnz_H / (sector_dim**2)
    print(
        f"Done. nnz={nnz_H}, sparsity={round(sparsity, 6)}. "
        f"t={round(elapsed, 3)}s"
    )
    return H
# ═════════════════════════════════════════════════════════════════════════════
# 13. Diagonalization helpers
# ═════════════════════════════════════════════════════════════════════════════


def diagonalize_block_dense(
    H: sp.csr_matrix, *, nev: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a small sparse block densely.

    Julia counterpart: ``diagonalize_block_dense``.  The dense Hermitian
    matrix is formed explicitly and diagonalized with
    :func:`numpy.linalg.eigh` (LAPACK ``zheevr``).

    Args:
        H: sparse Hermitian block.
        nev: number of lowest eigenvalues/eigenvectors to return.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(vals, vecs)`` — the ``k =
        min(nev, n)`` lowest eigenvalues (ascending) and their
        eigenvectors as the columns of an ``n × k`` matrix.
    """
    n = H.shape[0]
    if n == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0), dtype=np.complex128),
        )
    # Faithful to the Julia reference: `eigen(Hermitian(Matrix(H)))` runs
    # LAPACK zheev on the UPPER triangle of the raw block (for flux-aware
    # groups the projected block can carry complex self-loop diagonals in
    # this gauge; both languages treat it identically).
    H_dense = H.toarray()
    vals, vecs = np.linalg.eigh(H_dense, UPLO="U")
    k = min(nev, len(vals))
    return vals[:k].real, vecs[:, :k]


def diagonalize_block_arpack(
    H: sp.csr_matrix, *, nev: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a sparse block (dense for small, ARPACK for large).

    Julia counterpart: ``diagonalize_block_arpack`` (Arpack ``eigs`` with
    ``which=:SR`` for :math:`n > 500`; dense ``eigen`` otherwise).  The
    Python port uses :func:`scipy.sparse.linalg.eigsh` with ``which="SA"``
    (smallest algebraic), returning eigenvalues sorted ascending.

    Args:
        H: sparse Hermitian block.
        nev: number of lowest eigenvalues/eigenvectors to return.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(vals, vecs)``.
    """
    n = H.shape[0]
    if n == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0), dtype=np.complex128),
        )
    if n <= 500:
        return diagonalize_block_dense(H, nev=nev)
    # Faithful to the Julia reference `Arpack.eigs(H; which=:SR)`: a
    # GENERAL (non-Hermitian) ARPACK solve for the smallest real part on
    # the raw block, sorted ascending (the block may be non-Hermitian in
    # the flux-aware gauge; both languages treat it identically).
    vals, vecs = spla.eigs(H, k=nev, which="SR", tol=1e-10)
    order = np.argsort(vals.real)
    return vals.real[order], vecs[:, order]


def diagonalize_block_matrixfree(
    H_op: MatrixFreeHamiltonian, n: int, *, nev: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize using matrix-free Lanczos (ARPACK on a LinearOperator).

    Julia counterpart: ``diagonalize_block_matrixfree`` (KrylovKit
    ``eigsolve``).  For :math:`n \\le 500` the dense matrix is formed by
    applying ``H_op`` to the unit basis; for larger sectors ARPACK is
    driven through a :class:`scipy.sparse.linalg.LinearOperator`.

    Args:
        H_op: the matrix-free Hamiltonian operator.
        n: sector dimension.
        nev: number of lowest eigenvalues/eigenvectors to return.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(vals, vecs)``.
    """
    if n == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0), dtype=np.complex128),
        )
    if n <= 500:
        # Build the dense matrix from the unit basis (mirrors Julia) and
        # diagonalize on the UPPER triangle of the raw block.
        H_dense = np.empty((n, n), dtype=np.complex128)
        x = np.zeros(n, dtype=np.complex128)
        for j in range(n):
            x[j] = 1.0
            y = H_op(x)
            H_dense[:, j] = y
            x[j] = 0.0
        vals, vecs = np.linalg.eigh(H_dense, UPLO="U")
        k = min(nev, len(vals))
        return vals[:k].real, vecs[:, :k]

    def matvec(x: np.ndarray) -> np.ndarray:
        return np.asarray(H_op(x))

    linop = spla.LinearOperator(
        (n, n), matvec=matvec, dtype=np.complex128
    )
    rng = np.random.default_rng(12345)
    x0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x0 /= np.linalg.norm(x0)
    # Faithful to the Julia reference `KrylovKit.eigsolve(H_op, x0, nev,
    # :SR)`: a general (non-Hermitian) smallest-real-part solve on the raw
    # operator, sorted ascending.
    vals, vecs = spla.eigs(linop, k=nev, which="SR", v0=x0, tol=1e-10)
    order = np.argsort(vals.real)
    return vals.real[order], vecs[:, order]


# ═════════════════════════════════════════════════════════════════════════════
# 14. High-level ED data structure
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class Symmetry_Resolved_ED_Data:
    """All data for a symmetry-resolved ED run.

    Julia counterpart: ``mutable struct Symmetry_Resolved_ED_Data <:
    ED_Data``.

    Attributes:
        second_quantized_model: the model containing the lattice and the
            Hamiltonian parameters.
        n_filled: number of filled particles (set bits in the occupation
            basis).
        filling_fraction: the filling fraction ``n_filled / n_site`` per
            *flattened* vertex, as a :class:`fractions.Fraction`.
        symmetry_group: the symmetry group used for symmetry resolution.
        irrep_list: the list of 1-D irreps to resolve.
        orbit_catalog: the catalog of symmetry orbits.
        sector_dims: dimension of each symmetry sector (number of
            representative masks in the sector basis of each irrep).
        ed_scan_res: ``dict[irrep_idx → (eigvals, eigvecs)]`` storing the
            ED results of each scanned sector (``irrep_idx`` is a **0-based**
            index into ``irrep_list``).
    """

    second_quantized_model: Real_Space_Second_Quantized_Model
    n_filled: int
    filling_fraction: Fraction
    symmetry_group: Finite_Symmetry_Group
    irrep_list: list[OneDim_Irrep]
    orbit_catalog: Symmetry_Orbit_Catalog
    sector_dims: list[int]
    ed_scan_res: dict[int, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )


# ═════════════════════════════════════════════════════════════════════════════
# 15 & 16. ED scan — matrix and matrix-free modes
# ═════════════════════════════════════════════════════════════════════════════


def _find_irrep_idx(ed_data: Symmetry_Resolved_ED_Data, irrep_label) -> int:
    """**0-based** index of the irrep with the given label.

    Args:
        ed_data: the ED data structure.
        irrep_label: the label to find.

    Returns:
        int: the index, or ``-1`` if absent (Julia asserts non-nothing).

    Raises:
        ValueError: if no irrep carries the label.
    """
    for idx, irrep in enumerate(ed_data.irrep_list):
        if irrep.label == irrep_label:
            return idx
    raise ValueError(f"No irrep with label {irrep_label!r} in irrep_list.")


def ed_scan_at_irrep_matrix(
    irrep_label, ed_data: Symmetry_Resolved_ED_Data, *, nev: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize one irrep sector in matrix mode and store the result.

    Julia counterpart: ``ed_scan_at_irrep_matrix!``.

    Args:
        irrep_label: label of the target irrep.
        ed_data: the ED data structure (``ed_scan_res`` is updated).
        nev: number of lowest eigenvalues/eigenvectors to compute.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(vals, vecs)`` of the sector.
    """
    irrep_idx = _find_irrep_idx(ed_data, irrep_label)
    irrep = ed_data.irrep_list[irrep_idx]
    particle_statistics = ed_data.second_quantized_model.particle_statistics
    basis = build_symmetry_sector_basis(ed_data.orbit_catalog, irrep)
    ed_data.sector_dims[irrep_idx] = len(basis.representative_mask_list)

    cmap = CanonicalMap(ed_data.symmetry_group, particle_statistics, ed_data.orbit_catalog)

    H = build_ed_Hamiltonian_symmetry_block(
        basis,
        ed_data.second_quantized_model.bilinear_terms,
        ed_data.second_quantized_model.density_density_terms,
        cmap,
    )
    vals, vecs = diagonalize_block_arpack(H, nev=nev)
    ed_data.ed_scan_res[irrep_idx] = (vals, vecs)
    del H, cmap
    return vals, vecs


def ed_scan_at_irrep_matrixfree(
    irrep_label,
    ed_data: Symmetry_Resolved_ED_Data,
    *,
    nev: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize one irrep sector in matrix-free mode and store the result.

    Julia counterpart: ``ed_scan_at_irrep_matrixfree!``.

    Args:
        irrep_label: label of the target irrep.
        ed_data: the ED data structure (``ed_scan_res`` is updated).
        nev: number of lowest eigenvalues/eigenvectors to compute.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(vals, vecs)`` of the sector.
    """
    irrep_idx = _find_irrep_idx(ed_data, irrep_label)
    irrep = ed_data.irrep_list[irrep_idx]
    particle_statistics = ed_data.second_quantized_model.particle_statistics
    basis = build_symmetry_sector_basis(ed_data.orbit_catalog, irrep)
    sector_dim = len(basis.representative_mask_list)
    ed_data.sector_dims[irrep_idx] = sector_dim

    if sector_dim == 0:
        ed_data.ed_scan_res[irrep_idx] = (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0), dtype=np.complex128),
        )
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0), dtype=np.complex128),
        )

    print(
        f"\tMatrix-free mode @ irrep {irrep.label!r} (dim={sector_dim}, "
        f"numpy-vectorized) ... ",
        end="",
        flush=True,
    )

    bilinear = ed_data.second_quantized_model.bilinear_terms
    density = ed_data.second_quantized_model.density_density_terms

    import time

    t0 = time.perf_counter()
    cmap = CanonicalMap(ed_data.symmetry_group, particle_statistics, ed_data.orbit_catalog)
    populate_canonical_map(cmap, basis, bilinear)
    H_op, n = hamiltonian_linear_operator(basis, bilinear, density, cmap)
    vals, vecs = diagonalize_block_matrixfree(H_op, n, nev=nev)
    elapsed = time.perf_counter() - t0
    print(f"Done. t={round(elapsed, 3)}s")

    ed_data.ed_scan_res[irrep_idx] = (vals, vecs)
    del cmap, H_op
    return vals, vecs


def _ed_scan_sectors(
    ed_data: Symmetry_Resolved_ED_Data,
    *,
    nev: int = 5,
    mode: str = "matrix",
    checkpoint_path: str | None = None,
    scanned_sectors=None,
    overwrite: bool = False,
) -> str | None:
    """Internal: scan the sectors of ``ed_data`` (with checkpoint resume).

    Args:
        ed_data: the ED data structure.
        nev: eigenvalues per sector.
        mode: ``"matrix"`` or ``"matrixfree"``.
        checkpoint_path: optional checkpoint file; if it exists and
            ``overwrite`` is ``False``, resume from it.
        scanned_sectors: optional list of sector labels to restrict the scan.
        overwrite: whether to ignore an existing checkpoint.

    Returns:
        str | None: the checkpoint path if one was used, else ``None``.
    """
    if checkpoint_path is not None and os.path.isfile(checkpoint_path) and not overwrite:
        print(f"[ED scan] Loading existing checkpoint: {checkpoint_path}")
        loaded = load_checkpoint(checkpoint_path)
        ed_data.ed_scan_res = loaded.ed_scan_res
        return checkpoint_path

    n_total = len(ed_data.irrep_list)
    n_done = len(ed_data.ed_scan_res)
    if n_done > 0:
        print(
            f"[ED scan] {n_done}/{n_total} sectors already computed; "
            "resuming."
        )

    for irrep_idx, irrep in enumerate(ed_data.irrep_list):
        if irrep_idx in ed_data.ed_scan_res:
            continue
        if scanned_sectors is not None and irrep.label not in scanned_sectors:
            continue
        print(
            f"[ED scan] Sector {irrep_idx + 1}/{n_total} — irrep "
            f"{irrep.label!r}  [mode={mode}]",
            flush=True,
        )
        if mode == "matrixfree":
            ed_scan_at_irrep_matrixfree(
                irrep.label, ed_data, nev=nev
            )
        elif mode == "matrix":
            ed_scan_at_irrep_matrix(irrep.label, ed_data, nev=nev)
        else:
            raise ValueError(
                f"Unknown ED scan mode: {mode}. Use 'matrix' or 'matrixfree'."
            )
        if checkpoint_path is not None:
            save_checkpoint(ed_data, checkpoint_path)
            print(f"    [checkpoint → {checkpoint_path}]")
    return checkpoint_path


# ═════════════════════════════════════════════════════════════════════════════
# 17d. Checkpoint support
# ═════════════════════════════════════════════════════════════════════════════


def save_checkpoint(ed_data: Symmetry_Resolved_ED_Data, path: str) -> None:
    """Save ED data atomically to a checkpoint file (pickle).

    Julia counterpart: ``save_checkpoint`` (JLD2).  One job owns each
    checkpoint path; a fixed sibling temporary file prevents stale files
    from accumulating and preserves the previous checkpoint on failure.

    Args:
        ed_data: the ED data structure to save.
        path: destination ``.pkl`` path (parent directory is created).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "wb") as f:
            pickle.dump(ed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, path)
    finally:
        if os.path.isfile(temp_path):
            os.remove(temp_path)


def load_checkpoint(path: str) -> Symmetry_Resolved_ED_Data:
    """Load ED data from a checkpoint file.

    Args:
        path: the ``.pkl`` checkpoint file.

    Returns:
        Symmetry_Resolved_ED_Data: the restored data structure.
    """
    with open(path, "rb") as f:
        ed_data = pickle.load(f)
    return ed_data


# ═════════════════════════════════════════════════════════════════════════════
# 17e. Flux-insertion checkpoint filename
# ═════════════════════════════════════════════════════════════════════════════


def ed_scan_checkpoint_filename(
    model: Real_Space_Second_Quantized_Model,
    twisted_phases_over_2π: list[float],
    filling_fraction: Fraction,
) -> str:
    """Generate the universal checkpoint filename for an ED scan.

    Julia counterpart: ``ed_scan_checkpoint_filename``.  The format reads
    ``{tb_model.model_name}_{sample_size}_ν_graph={num}_{den}_twisted_phases_over_2π=
    {twisted_phases_over_2π}_params={params_short}.pkl``; ``params_short``
    rounds ALL values of ``params`` to 3 digits.

    Args:
        model: the second-quantized model.
        twisted_phases_over_2π: twisted phases used in the scan.
        filling_fraction: filling fraction per flattened vertex.

    Returns:
        str: the checkpoint filename (extension ``.pkl``).
    """
    params_short = {
        k: round(v, 3) if isinstance(v, (int, float)) else v
        for k, v in model.params.items()
    }
    return (
        f"{model.tb_model.model_name}_{model.lattice.sample_size}_ν_graph="
        f"{filling_fraction.numerator}_{filling_fraction.denominator}_"
        f"twisted_phases_over_2π={twisted_phases_over_2π}_"
        f"params={params_short}.pkl"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Unified ED scan API
# ═════════════════════════════════════════════════════════════════════════════


def ed_scan(
    ed_data: Symmetry_Resolved_ED_Data,
    *,
    nev: int = 5,
    mode: str = "matrix",
    scanned_sectors=None,
    checkpoint_path: str | None = None,
    checkpoint_dir: str = "checkpoints",
    overwrite: bool = False,
    flux_direction: int = 1,
    twisted_phases_over_2π_list: list[float] | None = None,
) -> list[str] | None:
    """Unified API for the ED scan, with flux scan and checkpoint resume.

    Julia counterpart: ``ed_scan!``.

    Args:
        ed_data: the ED data structure.
        nev: number of eigenvalues/eigenvectors to compute per sector.
        mode: the ED mode — ``"matrix"`` or ``"matrixfree"``.
        scanned_sectors: optional filter for specific sector labels.
        checkpoint_path: path of the checkpoint file to resume from; if it
            exists, scanning resumes from the last saved state.
        checkpoint_dir: directory for per-flux checkpoints in flux-scan mode.
        overwrite: whether to recompute sectors even if a checkpoint exists.
        flux_direction: for flux-insertion scans, the direction of the
            twisted boundary condition (**1-based**).
        twisted_phases_over_2π_list: the twisted phase (flux) values
            [in units of 2π] to scan over. If ``None``, fall back to the
            conventional ED scan (at fixed twisted boundary conditions).

    Returns:
        list[str] | None: if ``twisted_phases_over_2π_list is None``, the
        single-element checkpoint-path list (or ``None`` if no checkpoint
        was requested); otherwise the list of per-flux checkpoint paths.
    """
    if mode not in ("matrix", "matrixfree"):
        raise ValueError(f"Unknown ED scan mode: {mode}.")

    if twisted_phases_over_2π_list is None:  # conventional ED scan
        ckpt_path = _ed_scan_sectors(
            ed_data,
            nev=nev,
            mode=mode,
            checkpoint_path=checkpoint_path,
            scanned_sectors=scanned_sectors,
            overwrite=overwrite,
        )
        return None if ckpt_path is None else [ckpt_path]

    # ── Flux-scan mode ──
    model = ed_data.second_quantized_model
    filling_fraction = ed_data.filling_fraction
    os.makedirs(checkpoint_dir, exist_ok=True)
    dim = model.lattice.dim

    checkpoint_paths: list[str] = []
    for θ_val in twisted_phases_over_2π_list:
        flux = [0.0] * dim
        flux[flux_direction - 1] = float(θ_val)

        ckpt_name = ed_scan_checkpoint_filename(model, flux, filling_fraction)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

        if os.path.isfile(ckpt_path) and not overwrite:
            print(
                f"Skipping θ={θ_val} — checkpoint exists: {ckpt_name}"
            )
            checkpoint_paths.append(ckpt_path)
            continue

        from .second_quantized_model import (
            update_second_quantized_model_with_twisted_phases,
        )

        update_second_quantized_model_with_twisted_phases(
            model, twisted_phases_over_2π=flux
        )
        active_group = build_translation_group(model.lattice, flux)
        θ_ed_data = build_ed_data(
            model,
            filling_fraction=filling_fraction,
            symmetry_group=active_group,
        )
        _ed_scan_sectors(
            θ_ed_data,
            nev=nev,
            mode=mode,
            scanned_sectors=scanned_sectors,
            overwrite=overwrite,
        )
        save_checkpoint(θ_ed_data, ckpt_path)
        print(f"Saved ED scan @ θ={θ_val} → {ckpt_name}")
        checkpoint_paths.append(ckpt_path)

    return checkpoint_paths


# ═════════════════════════════════════════════════════════════════════════════
# 18. Pre-built symmetry groups
# ═════════════════════════════════════════════════════════════════════════════


def build_identity_group(n_site: int) -> Finite_Symmetry_Group:
    """Build the trivial one-element identity group.

    Args:
        n_site: number of vertices in the graph.

    Returns:
        Finite_Symmetry_Group: the group ``{"identity": [id]}``.
    """
    id_op = Symmetry_Operation("identity", list(range(1, n_site + 1)))
    return Finite_Symmetry_Group("identity", [id_op], identity_idx=0)


def build_translation_group(lattice, θ=None) -> Finite_Symmetry_Group:
    """Build the (flux-aware) lattice translation group.

    Julia counterpart: ``build_translation_group`` —
    :math:`\\mathbb{Z}^{L_1}\\times\\mathbb{Z}^{L_2}` (generalized to
    arbitrary ``dim`` sample sizes).

    When ``θ`` is omitted/empty/all-zeros, the operations are bare
    permutations (no per-site phases) — the fast path used for ordinary
    symmetry-resolved ED.

    When ``θ`` is supplied (or read from ``lattice.twisted_phases_over_2π``),
    each site that crosses a periodic boundary under translation acquires
    the gauge-covariant phase :math:`g(x)/g(Tx)` with
    :math:`g(x) = \\exp(i\\,2\\pi\\,\\theta\\cdot x/L)`.  The resulting group
    commutes with the flux-inserted Hamiltonian :math:`H(\\theta)` while
    keeping the irrep labels (ordinary momentum labels) unchanged — the
    entire flux physics is captured in the per-site phases of the group
    operations themselves.

    Args:
        lattice: the real-space lattice (``sample_size``, ``site_list``,
            ``site_to_index_map``, ``twisted_phases_over_2π``).
        θ: optional twisted phases; falls back to
            ``lattice.twisted_phases_over_2π``, then to zeros.

    Returns:
        Finite_Symmetry_Group: the translation group with operation labels
        ``(dx, dy, ...)``.
    """
    n_site = lattice.n_site
    L = lattice.sample_size

    # Resolve θ: explicit argument, then lattice built-in, then zeros
    if θ is None or len(θ) == 0:
        if not getattr(lattice, "twisted_phases_over_2π", []):
            θ_use = [0.0] * lattice.dim
        else:
            θ_use = [float(x) for x in lattice.twisted_phases_over_2π]
    else:
        if len(θ) != lattice.dim:
            raise ValueError(f"θ must have length {lattice.dim}.")
        θ_use = [float(x) for x in θ]

    Lf = [float(x) for x in L]

    ops: list[Symmetry_Operation] = []
    ranges = [range(n) for n in L]
    for shift_tuple in _product_ranges(ranges):
        shift = list(shift_tuple)
        perm = [0] * n_site
        phases = np.zeros(n_site, dtype=np.complex128)
        for (i, (cell_int, isub)) in enumerate(lattice.site_list):
            shifted_cell = tuple(
                (c + s) % Ld for c, s, Ld in zip(cell_int, shift, L)
            )
            perm[i] = lattice.site_to_index_map[(shifted_cell, isub)]
            phases[i] = cmath.exp(
                2j
                * math.pi
                * sum(
                    θd * (c - sc) / Ld
                    for θd, c, sc, Ld in zip(θ_use, cell_int, shifted_cell, Lf)
                )
            )
        ops.append(Symmetry_Operation(tuple(shift), perm, perm_phases=phases))
    return Finite_Symmetry_Group("translations", ops, identity_idx=0)


def _product_ranges(ranges: list[range]):
    """Ordered cartesian product of ranges (outer-most first).

    Julia counterpart: the nested ``for dx in ..., dy in ...`` loops of
    ``build_translation_group``.

    Args:
        ranges: list of ranges to iterate (outer-most first).

    Yields:
        tuple[int, ...]: one combination per iteration.
    """
    import itertools

    yield from itertools.product(*ranges)


# ═════════════════════════════════════════════════════════════════════════════
# 19. Pre-built irrep lists
# ═════════════════════════════════════════════════════════════════════════════


def build_identity_irrep_list() -> list[OneDim_Irrep]:
    """Trivial one-element irrep list for the identity group.

    Returns:
        list[OneDim_Irrep]: ``[OneDim_Irrep("identity", [1+0j])]``.
    """
    return [OneDim_Irrep("identity", [1.0 + 0.0j])]


def build_translation_irrep_list(
    G: Finite_Symmetry_Group, lattice
) -> list[OneDim_Irrep]:
    """Build the momentum irreps of the translation group.

    Julia counterpart: ``build_translation_irrep_list``.  The irrep labelled
    :math:`(k_1, k_2)` has the character

    .. math::
       \\chi_{(k_1,k_2)}(\\delta_1,\\delta_2) =
       e^{2\\pi i (k_1\\delta_1/L_1 + k_2\\delta_2/L_2)}.

    Args:
        G: the translation group (``name == "translations"``).
        lattice: the underlying real-space lattice.

    Returns:
        list[OneDim_Irrep]: one irrep per momentum ``(k1, k2)``, in
        ``k1``-major order.
    """
    assert G.name == "translations" and group_order(G) == _prod(lattice.sample_size)
    L1, L2 = lattice.sample_size
    irrep_list: list[OneDim_Irrep] = []
    for k1 in range(L1):
        for k2 in range(L2):
            chi_list = []
            for op in G.operations:
                dx, dy = op.label
                chi_list.append(
                    cmath.exp(
                        2j
                        * math.pi
                        * (k1 * dx / L1 + k2 * dy / L2)
                    )
                )
            irrep_list.append(OneDim_Irrep((k1, k2), chi_list))
    return irrep_list


def _prod(xs: list[int]) -> int:
    """Product of a list of integers.

    Args:
        xs: the integers.

    Returns:
        int: ``xs[0] * xs[1] * ...``.
    """
    out = 1
    for x in xs:
        out *= x
    return out


def build_irrep_list(G: Finite_Symmetry_Group, lattice) -> list[OneDim_Irrep]:
    """Dispatch to the irrep builder matching the group name.

    Args:
        G: the symmetry group.
        lattice: the underlying real-space lattice.

    Returns:
        list[OneDim_Irrep]: the irrep list.

    Raises:
        ValueError: if irrep construction for ``G.name`` is not implemented.
    """
    if G.name == "identity":
        return build_identity_irrep_list()
    elif G.name == "translations":
        return build_translation_irrep_list(G, lattice)
    else:
        raise ValueError(
            f"Irrep construction for group {G.name!r} is not yet implemented."
        )


# ═════════════════════════════════════════════════════════════════════════════
# 20. Convenience constructors
# ═════════════════════════════════════════════════════════════════════════════


def build_ed_data(
    second_quantized_model: Real_Space_Second_Quantized_Model,
    *,
    filling_fraction: Fraction | int | float = Fraction(1, 2),
    symmetry_group: Finite_Symmetry_Group,
) -> Symmetry_Resolved_ED_Data:
    """Core constructor of :class:`Symmetry_Resolved_ED_Data`.

    Julia counterpart: ``build_ed_data``.  Internally calls
    :func:`build_symmetry_orbit_catalog`, :func:`build_irrep_list`, etc.

    Args:
        second_quantized_model: the second-quantized model.
        filling_fraction: the filling **per flattened vertex** —
            ``n_filled = filling_fraction * n_site`` (exact rational
            arithmetic; a :class:`fractions.Fraction` or int/float).
        symmetry_group: the symmetry group used for the resolution.

    Returns:
        Symmetry_Resolved_ED_Data: the ED data structure.

    .. note::
        ``filling_fraction`` is the particle number per *flattened* vertex.
        For example, for a spinful Hubbard model with 2·(Lx·Ly) vertices,
        "half-filling" (one particle per site) means
        ``filling_fraction = Fraction(1, 2)``; for the bosonic Haldane
        honeycomb FCI at half filling of the *band*, the vertex filling is
        ``Fraction(1, 4)``.
    """
    n_site = second_quantized_model.lattice.n_site
    if not isinstance(filling_fraction, Fraction):
        filling_fraction = Fraction(filling_fraction)
    n_filled = int(filling_fraction * n_site)
    assert filling_fraction.denominator * n_filled == (
        filling_fraction.numerator * n_site
    ), (
        "build_ed_data: filling_fraction * n_site must be an integer "
        "(particle number)."
    )

    irrep_list = build_irrep_list(symmetry_group, second_quantized_model.lattice)
    catalog = build_symmetry_orbit_catalog(
        second_quantized_model=second_quantized_model,
        n_filled=n_filled,
        symmetry_group=symmetry_group,
        particle_statistics=second_quantized_model.particle_statistics,
    )
    sector_dims = [0] * len(irrep_list)

    return Symmetry_Resolved_ED_Data(
        second_quantized_model,
        n_filled,
        filling_fraction,
        symmetry_group,
        irrep_list,
        catalog,
        sector_dims,
    )


def full_ed(
    second_quantized_model: Real_Space_Second_Quantized_Model,
    n_filled: int,
    *,
    nev: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a full-Hilbert-space (identity-group) ED.

    Args:
        second_quantized_model: the second-quantized model.
        n_filled: particle number.
        nev: number of lowest eigenvalues/eigenvectors.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(vals, vecs)`` of the single
        identity sector.
    """
    symmetry = build_identity_group(second_quantized_model.lattice.n_site)
    ed_data = build_ed_data(
        second_quantized_model,
        filling_fraction=Fraction(
            n_filled, second_quantized_model.lattice.n_site
        ),
        symmetry_group=symmetry,
    )
    ed_scan(ed_data, nev=nev)
    return ed_data.ed_scan_res[0]


# ═════════════════════════════════════════════════════════════════════════════
# 21. Utility: print & plot spectrum
# ═════════════════════════════════════════════════════════════════════════════


def _spectrum_matrix(
    ed_data: Symmetry_Resolved_ED_Data, shift_to_zero: bool = True
) -> np.ndarray:
    """Assemble the ``(n_irrep × nev)`` spectrum matrix (NaN for missing).

    Args:
        ed_data: the ED data structure.
        shift_to_zero: whether to shift the spectrum so the minimum
            eigenvalue is zero.

    Returns:
        np.ndarray: the spectrum matrix.
    """
    scanned = sorted(ed_data.ed_scan_res.keys())
    if not scanned:
        return np.zeros((0, 0))
    n_irrep = len(ed_data.irrep_list)
    nev = max(len(ed_data.ed_scan_res[i][0]) for i in scanned)
    spec = np.full((n_irrep, nev), np.nan)
    for irrep_idx in scanned:
        vals = ed_data.ed_scan_res[irrep_idx][0]
        spec[irrep_idx, : len(vals)] = vals
    if shift_to_zero:
        finite = spec[~np.isnan(spec)]
        if finite.size:
            spec = spec - np.min(finite)
    return spec


def print_spectrum(
    ed_data: Symmetry_Resolved_ED_Data, *, shift_to_zero: bool = True
) -> np.ndarray:
    """Print the symmetry-resolved ED spectrum to stdout.

    Args:
        ed_data: the ED data structure.
        shift_to_zero: whether to shift the spectrum so the minimum
            eigenvalue is zero.

    Returns:
        np.ndarray: the spectrum matrix (rows = irrep, cols = eigenvalue #).
    """
    scanned = sorted(ed_data.ed_scan_res.keys())
    if not scanned:
        return np.zeros((0, 0))
    n_irrep = len(ed_data.irrep_list)
    nev = max(len(ed_data.ed_scan_res[i][0]) for i in scanned)
    spec = _spectrum_matrix(ed_data, shift_to_zero=shift_to_zero)
    print("ED Spectrum (rows = irrep, cols = eigenvalue #):")
    for irrep_idx in range(n_irrep):
        label = ed_data.irrep_list[irrep_idx].label
        vals_str = "  ".join(
            "  NaN" if math.isnan(spec[irrep_idx, e]) else f"{spec[irrep_idx, e]:.8f}"
            for e in range(nev)
        )
        print(f"  [{irrep_idx}] {label!r}: {vals_str}")
    return spec


def plot_spectrum(
    ed_data: Symmetry_Resolved_ED_Data, *, shift_to_zero: bool = True
) -> tuple[Any, Any]:
    """Plot the symmetry-resolved spectrum (matplotlib).

    Args:
        ed_data: the ED data structure.
        shift_to_zero: whether to shift the spectrum so the minimum
            eigenvalue is zero.

    Returns:
        tuple: ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    scanned = sorted(ed_data.ed_scan_res.keys())
    if not scanned:
        fig, ax = plt.subplots()
        return fig, ax
    n_irrep = len(ed_data.irrep_list)
    nev = max(len(ed_data.ed_scan_res[i][0]) for i in scanned)
    spec = _spectrum_matrix(ed_data, shift_to_zero=shift_to_zero)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set_xlabel("Irrep index")
    ax.set_ylabel("E")
    ax.set_title(
        f"ED Spectrum — {ed_data.second_quantized_model.lattice.sample_size}, "
        f"ν={ed_data.filling_fraction}"
    )
    for k in range(n_irrep):
        for e in range(nev):
            val = spec[k, e]
            if not math.isnan(val):
                ax.scatter(k, val, color="royalblue", s=28, alpha=0.75)
    fig.tight_layout()
    return fig, ax


def plot_ed_scan_res(
    ed_data: Symmetry_Resolved_ED_Data,
    *,
    shift_to_zero: bool = True,
    save_plot_path: str | None = None,
    show_spec: bool = True,
) -> tuple[Any, Any]:
    """Plot the ED spectrum resolved by symmetry sectors (1-D irreps).

    Julia counterpart: ``plot_ed_scan_res``.

    Args:
        ed_data: the ED data structure.
        shift_to_zero: whether to shift the spectrum such that the minimum
            eigenvalue is zero.
        save_plot_path: optional path to save the generated plot.
        show_spec: whether to print the numerical spectrum values.

    Returns:
        tuple: ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    scanned_indices = sorted(ed_data.ed_scan_res.keys())
    nev = (
        1
        if not scanned_indices
        else len(ed_data.ed_scan_res[scanned_indices[0]][0])
    )
    n_irrep = len(ed_data.irrep_list)
    spec = np.full((n_irrep, nev), np.nan)
    for irrep_idx in range(n_irrep):
        if irrep_idx in ed_data.ed_scan_res:
            vals = ed_data.ed_scan_res[irrep_idx][0]
            spec[irrep_idx, : len(vals)] = vals

    if show_spec:
        print("[ED] Raw ED spectrum (symmetry-resolved):")
        print(spec)

    finite_vals = spec[~np.isnan(spec)]
    if finite_vals.size and shift_to_zero:
        spec = spec - np.min(finite_vals)
    if show_spec:
        print("[ED] Shifted ED spectrum (symmetry-resolved):")
        print(spec)

    sample_size = ed_data.second_quantized_model.lattice.sample_size
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set_xlabel("symmetry sector index")
    ax.set_ylabel("E [unit]")
    ax.set_title(f"ED Spectrum of {sample_size} Sample")
    for k in range(spec.shape[0]):
        for e in range(spec.shape[1]):
            val = spec[k, e]
            if not math.isnan(val):
                ax.scatter(k, val, color="royalblue", s=28, alpha=0.75)
    fig.tight_layout()

    if save_plot_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_plot_path)), exist_ok=True)
        fig.savefig(save_plot_path)
        print(f"[ED] Saved plot to {save_plot_path}")

    return fig, ax


__all__ = [
    # data structures
    "Symmetry_Operation",
    "Finite_Symmetry_Group",
    "OneDim_Irrep",
    "Symmetry_Orbit_Catalog",
    "Symmetry_Sector_Basis",
    "Symmetry_Resolved_ED_Data",
    "CanonicalMap",
    "MatrixFreeHamiltonian",
    # group actions & canonicalization
    "apply_operation_to_mask",
    "get_canonical_representative",
    "get_canonical",
    "populate_canonical_map",
    "project_to_sector",
    "project_to_unnormalized_sector",
    "hopping_phase_for_stats",
    # builders
    "build_identity_group",
    "build_translation_group",
    "build_identity_irrep_list",
    "build_translation_irrep_list",
    "build_irrep_list",
    "build_symmetry_orbit_catalog",
    "update_orbit_stabilizer_phases",
    "build_symmetry_sector_basis",
    "build_ed_data",
    # Hamiltonian
    "build_ed_Hamiltonian_symmetry_block",
    "hamiltonian_linear_operator",
    "apply_hamiltonian",
    "apply_hamiltonian_with_threads",
    # ED scan & diagonalization
    "ed_scan",
    "ed_scan_at_irrep_matrix",
    "ed_scan_at_irrep_matrixfree",
    "diagonalize_block_dense",
    "diagonalize_block_arpack",
    "diagonalize_block_matrixfree",
    "full_ed",
    # checkpoint
    "save_checkpoint",
    "load_checkpoint",
    "ed_scan_checkpoint_filename",
    # presentation
    "print_spectrum",
    "plot_spectrum",
    "plot_ed_scan_res",
]
